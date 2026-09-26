import tempfile
from pathlib import Path
from backend.core.engine import ResearchEngine
from backend.core.state import ResearchPhase, ResearchStatus
from backend.core.limits import ResearchLimits
from backend.llm.mock import MockLLMBackend
from backend.db.database import DatabaseManager
from tests.conftest import FakeSearchTool, FakeFetchTool


def test_research_engine_plan_search_fetch():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_engine.db"
        db = DatabaseManager(db_path=db_path)

        canned = {
            "ResearchPlanSchema": {
                "goal": "Compare Model X vs Model Y",
                "tasks": [
                    {"task_id": "task_1", "description": "Find Model X NDS metric", "expected_evidence": "nuScenes NDS"},
                    {"task_id": "task_2", "description": "Find Model Y NDS metric", "expected_evidence": "nuScenes NDS"}
                ],
                "key_hypotheses": ["Model X beats Model Y"]
            },
            "GeneratedQueriesSchema": {
                "queries": [
                    {"query": "Model X nuScenes NDS", "query_type": "evidence", "rationale": "Direct benchmark search"}
                ]
            }
        }
        mock_llm = MockLLMBackend(canned_responses=canned)
        fake_search = FakeSearchTool()
        fake_fetch = FakeFetchTool()

        engine = ResearchEngine(
            llm=mock_llm,
            db=db,
            limits=ResearchLimits(max_research_steps=2),
            search_tool=fake_search,
            fetch_tool=fake_fetch
        )

        state = engine.create_session("Compare Model X vs Model Y")
        assert state.phase == ResearchPhase.INIT
        assert state.status == ResearchStatus.PENDING

        # Step 1: Plan
        engine.run_plan_phase(state)
        assert state.phase == ResearchPhase.PLAN
        assert state.plan is not None
        assert len(state.plan.tasks) == 2

        # Step 2: Search
        urls = engine.run_search_phase(state)
        assert state.phase == ResearchPhase.SEARCH
        assert len(urls) > 0
        assert len(state.source_ids) > 0

        # Step 3: Fetch
        docs = engine.run_fetch_phase(state, urls=urls)
        assert state.phase == ResearchPhase.FETCH
        assert len(docs) > 0
