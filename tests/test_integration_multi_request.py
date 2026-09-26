import pytest
from backend.core.state import ResearchPhase, ResearchStatus, OpenQuestion
from backend.core.limits import ResearchLimits
from backend.core.errors import BudgetExceededError, ModelInferenceError
from backend.core.budget import ExecutionBudgetTracker
from backend.llm.mock import MockLLMBackend


def test_plan_then_search_restores_full_state(client, isolated_engine):
    """
    P0-1 Integration Test:
    Ensures that when a user calls /plan, the plan is persisted,
    and when the user subsequently calls /search, the state is fully
    restored from DB so the query generator prompt contains the plan.
    Zero real network calls; isolated in temporary SQLite database.
    """
    canned = {
        "ResearchPlanSchema": {
            "goal": "Compare PointPillars and CenterPoint",
            "tasks": [
                {"task_id": "task_1", "description": "Extract PointPillars nuScenes mAP", "expected_evidence": "mAP"},
                {"task_id": "task_2", "description": "Extract CenterPoint nuScenes mAP", "expected_evidence": "mAP"}
            ],
            "key_hypotheses": ["CenterPoint outperforms PointPillars"]
        },
        "GeneratedQueriesSchema": {
            "queries": [
                {"query": "PointPillars vs CenterPoint nuScenes mAP", "query_type": "evidence", "rationale": "benchmark"}
            ]
        }
    }
    isolated_engine.llm = MockLLMBackend(canned_responses=canned)

    # 1. Create Session
    create_res = client.post("/api/research/session", json={"goal": "Compare PointPillars and CenterPoint"})
    assert create_res.status_code == 200
    session_id = create_res.json()["session_id"]

    # 2. Call /plan
    plan_res = client.post(f"/api/research/session/{session_id}/plan")
    assert plan_res.status_code == 200
    assert plan_res.json()["plan"] is not None
    assert len(plan_res.json()["plan"]["tasks"]) == 2

    # Clear mock LLM call history to inspect only subsequent calls
    isolated_engine.llm.call_history.clear()

    # 3. Call /search (simulating separate HTTP request where state must be restored from DB)
    search_res = client.post(f"/api/research/session/{session_id}/search")
    assert search_res.status_code == 200

    # 4. Verify search query generator received the restored plan in its prompt
    assert len(isolated_engine.llm.call_history) > 0
    query_call = isolated_engine.llm.call_history[0]
    prompt_sent = query_call["prompt"]

    assert "PointPillars" in prompt_sent
    assert "CenterPoint" in prompt_sent
    assert "Extract PointPillars nuScenes mAP" in prompt_sent


def test_end_to_end_run_api(client, isolated_engine):
    """
    P0-3, P0-4, P0-5 & Step Enforcement Integration Test:
    Ensures POST /api/research/run executes Question -> Plan -> Search -> Fetch -> Evaluate -> Answer -> Done
    with step >= 1 and persists report and budget in SQLite.
    """
    canned = {
        "ResearchPlanSchema": {
            "goal": "Benchmark FlashAttention-2 speedup",
            "tasks": [
                {"task_id": "t1", "description": "Find TFLOPs throughput", "expected_evidence": "TFLOPs"}
            ],
            "key_hypotheses": ["2x speedup over standard attention"]
        },
        "GeneratedQueriesSchema": {
            "queries": [
                {"query": "FlashAttention-2 benchmark A100", "query_type": "evidence", "rationale": "Throughput"}
            ]
        }
    }
    isolated_engine.llm = MockLLMBackend(canned_responses=canned)

    run_res = client.post("/api/research/run", json={"goal": "Benchmark FlashAttention-2 speedup"})
    assert run_res.status_code == 200
    data = run_res.json()

    assert data["status"] == "COMPLETED"
    assert data["phase"] == "DONE"
    assert data["step"] >= 1  # Verifies real EVALUATE loop execution!
    assert data["plan"] is not None
    assert data["answer"] != ""
    assert "budget" in data
    assert "steps" in data["budget"]

    # Verify session retrieval endpoint includes report
    session_id = data["session_id"]
    get_res = client.get(f"/api/research/session/{session_id}")
    assert get_res.status_code == 200
    body = get_res.json()
    assert body["report"] is not None
    assert body["report"]["content_markdown"] == data["answer"]

    # Trajectory verification: verify trajectory is placed in raw partition with verified=False (resolves P0 fake confidence)
    trajs = isolated_engine.trajectory_repo.list_by_partition("raw")
    assert len(trajs) >= 1
    answer_traj = next((t for t in trajs if t["task_type"] == "answer_synthesis"), None)
    assert answer_traj is not None
    assert answer_traj["verified"] == 0  # Not verified yet in Week 1!


def test_error_handling_and_status_codes(client, isolated_engine):
    """
    P1 Tests for HTTP Status Code Mappings and Robustness:
    - 409 Conflict on invalid state transition (calling /plan on DONE session)
    - 404 Not Found on unknown session
    - 502 Bad Gateway on ModelInferenceError
    """
    # 1. 404 test
    res_404 = client.get("/api/research/session/sess_nonexistent_999")
    assert res_404.status_code == 404

    # 2. 409 Conflict test
    create_res = client.post("/api/research/session", json={"goal": "State machine test"})
    sid = create_res.json()["session_id"]
    # Mark session as DONE
    state = isolated_engine.load_state(sid)
    state.phase = ResearchPhase.DONE
    state.status = ResearchStatus.COMPLETED
    isolated_engine.save_state(state)

    # Calling /plan on a DONE session must return 409 Conflict
    conflict_res = client.post(f"/api/research/session/{sid}/plan")
    assert conflict_res.status_code == 409
    assert "StateTransitionError" in conflict_res.json()["error"]

    # 3. 502 Bad Gateway test on ModelInferenceError
    class FailingLLM(MockLLMBackend):
        def structured_generate(self, *args, **kwargs):
            raise ModelInferenceError("Ollama connection timed out")

    isolated_engine.llm = FailingLLM()
    create_res2 = client.post("/api/research/session", json={"goal": "Failing model test"})
    sid2 = create_res2.json()["session_id"]

    fail_res = client.post(f"/api/research/session/{sid2}/plan")
    assert fail_res.status_code == 502
    assert "ModelInferenceError" in fail_res.json()["error"]

    # Session in DB must be transitioned to FAILED, not left dangling in PENDING
    st_failed = isolated_engine.load_state(sid2)
    assert st_failed.status == ResearchStatus.FAILED
    assert "Ollama connection timed out" in (st_failed.error_message or "")


def test_budget_limits_enforce_hard_stops():
    """
    P0-2 Test:
    Verifies that ExecutionBudgetTracker strictly blocks calls when limits are reached.
    """
    limits = ResearchLimits(max_search_calls=2, max_llm_calls=2, max_runtime_seconds=1)
    tracker = ExecutionBudgetTracker(limits=limits)

    tracker.record_search()
    tracker.record_search()
    assert not tracker.can_search()

    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.record_search()
    assert "Maximum search calls limit reached" in str(exc_info.value)

    tracker.record_llm_call(tokens=100)
    tracker.record_llm_call(tokens=200)
    assert not tracker.can_call_llm()

    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.record_llm_call(tokens=50)
    assert "Maximum LLM calls limit reached" in str(exc_info.value)


def test_multi_step_e2e_terminates_at_max_steps(isolated_engine):
    """
    Verifies that run_week1 loops through EVALUATE when open questions exist
    and stops strictly when state.step reaches limits.max_research_steps without infinite loop.
    """
    isolated_engine.limits.max_research_steps = 2
    state = isolated_engine.create_session("Multi-step loop test")
    state.open_questions = [OpenQuestion(question_id="q1", text="What is the latency?")]
    isolated_engine.save_state(state)

    result = isolated_engine.run_week1(state)
    assert result["step"] == 2  # Proves loop ran 2 times and stopped cleanly at max_research_steps!
    assert result["status"] in {"COMPLETED", "PARTIAL"}

