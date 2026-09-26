import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app, research_engine
from backend.llm.mock import MockLLMBackend
from backend.core.limits import ResearchLimits
from backend.core.errors import BudgetExceededError
from backend.core.budget import ExecutionBudgetTracker


def test_plan_then_search_restores_full_state():
    """
    P0-1 Integration Test:
    Ensures that when a user calls /plan, the plan is persisted,
    and when the user subsequently calls /search, the state is fully
    restored from DB so the query generator prompt contains the plan.
    """
    client = TestClient(app)

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
    mock_llm = MockLLMBackend(canned_responses=canned)
    # Inject mock LLM into the engine
    research_engine.llm = mock_llm

    # 1. Create Session
    create_res = client.post("/api/research/session", json={"goal": "Compare PointPillars and CenterPoint"})
    assert create_res.status_code == 200
    session_id = create_res.json()["session_id"]

    # 2. Call /plan
    plan_res = client.post(f"/api/research/session/{session_id}/plan")
    assert plan_res.status_code == 200
    assert plan_res.json()["plan"] is not None
    assert len(plan_res.json()["plan"]["tasks"]) == 2

    # Clear mock LLM call history
    mock_llm.call_history.clear()

    # 3. Call /search (simulating separate HTTP request where in-memory state must be restored from DB)
    search_res = client.post(f"/api/research/session/{session_id}/search")
    assert search_res.status_code == 200

    # 4. Verify search query generator received the restored plan in its prompt
    assert len(mock_llm.call_history) > 0
    query_call = mock_llm.call_history[0]
    prompt_sent = query_call["prompt"]

    # P0 verification: prompt MUST contain the saved plan details
    assert "PointPillars" in prompt_sent
    assert "CenterPoint" in prompt_sent
    assert "Extract PointPillars nuScenes mAP" in prompt_sent


def test_end_to_end_run_api():
    """
    P0-3, P0-4, P0-5 Integration Test:
    Ensures POST /api/research/run executes Question -> Plan -> Search -> Fetch -> Answer -> Done
    and persists report in SQLite.
    """
    client = TestClient(app)

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
    mock_llm = MockLLMBackend(canned_responses=canned)
    research_engine.llm = mock_llm

    from tests.test_engine import FakeSearchTool, FakeFetchTool
    research_engine.search_tool = FakeSearchTool()
    research_engine.fetch_tool = FakeFetchTool()

    run_res = client.post("/api/research/run", json={"goal": "Benchmark FlashAttention-2 speedup"})
    assert run_res.status_code == 200
    data = run_res.json()

    assert data["status"] == "COMPLETED"
    assert data["phase"] == "DONE"
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


def test_budget_limits_enforce_hard_stops():
    """
    P0-2 Test:
    Verifies that ExecutionBudgetTracker strictly blocks calls when limits are reached.
    """
    limits = ResearchLimits(max_search_calls=2, max_llm_calls=2, max_runtime_seconds=1)
    tracker = ExecutionBudgetTracker(limits=limits)

    # Search calls
    tracker.record_search()
    tracker.record_search()
    assert not tracker.can_search()

    try:
        tracker.record_search()
        assert False, "Should have raised BudgetExceededError"
    except BudgetExceededError as e:
        assert "Maximum search calls limit reached" in str(e)

    # LLM calls
    tracker.record_llm_call(tokens=100)
    tracker.record_llm_call(tokens=200)
    assert not tracker.can_call_llm()

    try:
        tracker.record_llm_call(tokens=50)
        assert False, "Should have raised BudgetExceededError"
    except BudgetExceededError as e:
        assert "Maximum LLM calls limit reached" in str(e)
