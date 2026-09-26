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


def test_invalid_transition_does_not_corrupt_db_session(client, isolated_engine):
    """
    Regression Test for P0 Bug: _handle_phase_error corrupting valid session.
    1. Calling /plan on a session currently in SEARCH must return 409 Conflict AND keep phase SEARCH (not PARTIAL/FAILED).
    2. Calling /run with session_id of an already DONE session must return 409 Conflict AND keep phase DONE.
    """
    # 1. Create and move session to SEARCH
    canned = {
        "ResearchPlanSchema": {
            "goal": "Test invariant protection",
            "tasks": [{"task_id": "t1", "description": "task 1", "expected_evidence": "ev1"}],
            "key_hypotheses": ["hyp1"]
        },
        "GeneratedQueriesSchema": {
            "queries": [{"query": "test query", "query_type": "evidence", "rationale": "r1"}]
        }
    }
    isolated_engine.llm = MockLLMBackend(canned_responses=canned)

    create_res = client.post("/api/research/session", json={"goal": "Test invariant protection"})
    sid = create_res.json()["session_id"]

    plan_res = client.post(f"/api/research/session/{sid}/plan")
    assert plan_res.status_code == 200

    search_res = client.post(f"/api/research/session/{sid}/search")
    assert search_res.status_code == 200

    # Verify session is currently in SEARCH
    st_search = isolated_engine.load_state(sid)
    assert st_search.phase == ResearchPhase.SEARCH

    # Now illegally call /plan again on the SEARCH session
    invalid_plan_res = client.post(f"/api/research/session/{sid}/plan")
    assert invalid_plan_res.status_code == 409

    # CRITICAL: Verify DB state was NOT corrupted to PARTIAL or FAILED!
    st_after = isolated_engine.load_state(sid)
    assert st_after.phase == ResearchPhase.SEARCH
    assert st_after.status == ResearchStatus.RUNNING

    # 2. Test calling /run on an already DONE session
    st_after.phase = ResearchPhase.DONE
    st_after.status = ResearchStatus.COMPLETED
    isolated_engine.save_state(st_after)

    invalid_run_res = client.post("/api/research/run", json={"session_id": sid})
    assert invalid_run_res.status_code == 409

    # CRITICAL: Terminal state invariant holds; remains DONE, not mutated to PARTIAL
    st_done = isolated_engine.load_state(sid)
    assert st_done.phase == ResearchPhase.DONE
    assert st_done.status == ResearchStatus.COMPLETED


def test_run_e2e_failing_llm_returns_502(client, isolated_engine):
    """
    Regression Test for P1 Bug: run_week1 returning HTTP 200 with {'error': ...}.
    When LLM fails, POST /api/research/run must propagate exception so FastAPI returns HTTP 502.
    """
    class FailingLLM(MockLLMBackend):
        def structured_generate(self, *args, **kwargs):
            raise ModelInferenceError("Ollama crashed during inference")

    isolated_engine.llm = FailingLLM()
    res = client.post("/api/research/run", json={"goal": "Test failing model in /run"})
    assert res.status_code == 502
    assert "ModelInferenceError" in res.json()["error"]


def test_budget_step_sync(client, isolated_engine):
    """
    Regression Test for P1 Bug: budget.step_count not synchronized with state.step.
    """
    canned = {
        "ResearchPlanSchema": {
            "goal": "Test step sync",
            "tasks": [{"task_id": "t1", "description": "sync step", "expected_evidence": "ev"}],
            "key_hypotheses": ["h1"]
        },
        "GeneratedQueriesSchema": {
            "queries": [{"query": "query sync", "query_type": "evidence", "rationale": "r"}]
        }
    }
    isolated_engine.llm = MockLLMBackend(canned_responses=canned)

    res = client.post("/api/research/run", json={"goal": "Test step sync"})
    assert res.status_code == 200
    data = res.json()

    assert data["step"] >= 1
    assert data["budget"]["steps"] == data["step"]

    # Verify tracker loaded from DB matches state.step
    tracker = isolated_engine.get_budget_tracker(data["session_id"])
    assert tracker.step_count == data["step"]


def test_run_with_nonexistent_session_id_returns_404(client):
    """
    Verifies that calling /run with a nonexistent session_id returns 404 Not Found.
    """
    res = client.post("/api/research/run", json={"session_id": "sess_nonexistent_99999"})
    assert res.status_code == 404




def test_error_between_phases_does_not_leave_session_hanging(isolated_engine):
    """A failure outside phase methods (e.g. in EVALUATE) must still move the session to a terminal state."""
    def _broken_evaluate(state):
        raise RuntimeError("evaluate bug")
    isolated_engine.state_machine.evaluate_next_step = _broken_evaluate
    isolated_engine.llm = MockLLMBackend(canned_responses={
        "GeneratedQueriesSchema": {"queries": [{"query": "latency benchmark", "rationale": "r"}]}
    })

    state = isolated_engine.create_session("Evaluate failure test")
    with pytest.raises(RuntimeError, match="evaluate bug"):
        isolated_engine.run_week1(state)

    restored = isolated_engine.load_state(state.session_id)
    assert restored.phase == ResearchPhase.PARTIAL  # sources were already collected
    assert restored.status == ResearchStatus.PARTIAL
    assert restored.error_message == "evaluate bug"


def test_error_message_can_be_cleared(isolated_engine):
    state = isolated_engine.create_session("Clear error test")
    state.error_message = "transient failure"
    isolated_engine.save_state(state)
    assert isolated_engine.load_state(state.session_id).error_message == "transient failure"

    state.error_message = None
    isolated_engine.save_state(state)
    assert isolated_engine.load_state(state.session_id).error_message is None
