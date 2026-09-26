import uuid
import logging
from typing import Optional, List
from backend.core.state import ResearchState, ResearchPhase, ResearchStatus, ResearchPlan, PlanTask, OpenQuestion
from backend.core.limits import ResearchLimits
from backend.core.budget import ExecutionBudgetTracker
from backend.core.transitions import StateMachine
from backend.core.errors import BudgetExceededError, StateTransitionError
from backend.llm.backend import LLMBackend
from backend.llm.schemas import ResearchPlanSchema, GeneratedQueriesSchema
from backend.db.database import DatabaseManager
from backend.db.repositories import SessionRepository, SourceRepository, RawEvidenceRepository, TrajectoryRepository
from backend.tools.web_search import WebSearchTool
from backend.tools.web_fetch import WebFetchTool

logger = logging.getLogger(__name__)


class ResearchEngine:
    def __init__(
        self,
        llm: LLMBackend,
        db: DatabaseManager,
        limits: Optional[ResearchLimits] = None,
        search_tool: Optional[WebSearchTool] = None,
        fetch_tool: Optional[WebFetchTool] = None
    ):
        self.llm = llm
        self.db = db
        self.limits = limits or ResearchLimits()
        self.state_machine = StateMachine(limits=self.limits)
        self.search_tool = search_tool or WebSearchTool(max_results=self.limits.max_search_results)
        self.fetch_tool = fetch_tool or WebFetchTool()

        # Repositories
        self.session_repo = SessionRepository(self.db)
        self.source_repo = SourceRepository(self.db)
        self.raw_evidence_repo = RawEvidenceRepository(self.db)
        self.trajectory_repo = TrajectoryRepository(self.db)

    def create_session(self, goal: str, session_id: Optional[str] = None) -> ResearchState:
        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        state = ResearchState(
            session_id=sid,
            goal=goal,
            phase=ResearchPhase.INIT,
            status=ResearchStatus.PENDING
        )
        self.session_repo.create(
            session_id=sid,
            goal=goal,
            phase=ResearchPhase.INIT.value,
            status=ResearchStatus.PENDING.value
        )
        logger.info(f"[ResearchEngine] Created session {sid} with goal: {goal}")
        return state

    def run_plan_phase(self, state: ResearchState) -> None:
        """Executes the PLAN phase using structured LLM generation."""
        self.state_machine.transition(state, ResearchPhase.PLAN, reason="Starting research plan")
        prompt = (
            f"You are a Senior Research Planner. Given the following research question, "
            f"break it down into 3-5 concrete, factual investigation sub-tasks and identify core hypotheses to test.\n\n"
            f"Question: {state.goal}"
        )
        res = self.llm.structured_generate(prompt, schema=ResearchPlanSchema)
        plan_data: ResearchPlanSchema = res.parsed

        tasks = [
            PlanTask(
                task_id=t.task_id,
                description=t.description,
                expected_evidence=t.expected_evidence
            )
            for t in plan_data.tasks
        ]
        state.plan = ResearchPlan(
            goal=state.goal,
            tasks=tasks,
            key_hypotheses=plan_data.key_hypotheses
        )

        # Sync session to DB
        self.session_repo.update_phase_and_step(
            session_id=state.session_id,
            phase=state.phase.value,
            step=state.step,
            status=state.status.value,
            plan_json=state.plan.model_dump_json()
        )

        # Save trajectory sample in raw partition
        self.trajectory_repo.add(
            trajectory_id=f"traj_{uuid.uuid4().hex[:12]}",
            session_id=state.session_id,
            task_type="planning",
            model_source=getattr(self.llm, "model", "mock"),
            payload={"goal": state.goal, "plan": state.plan.model_dump()},
            partition="raw"
        )
        logger.info(f"[ResearchEngine] Plan completed for session {state.session_id} with {len(tasks)} tasks.")

    def run_search_phase(self, state: ResearchState, custom_queries: Optional[List[str]] = None) -> List[str]:
        """Generates targeted search queries and executes web search."""
        self.state_machine.transition(state, ResearchPhase.SEARCH, reason="Executing web search")
        
        queries_to_run = custom_queries or []
        if not queries_to_run:
            prompt = (
                f"Based on the research goal and plan, generate 2-4 search queries across categories "
                f"(discovery, evidence, verification, contradiction) to gather required facts.\n\n"
                f"Goal: {state.goal}\n"
                f"Plan: {state.plan.model_dump_json() if state.plan else ''}\n"
                f"Previously visited: {state.visited_queries}"
            )
            res = self.llm.structured_generate(prompt, schema=GeneratedQueriesSchema)
            generated: GeneratedQueriesSchema = res.parsed
            queries_to_run = [q.query for q in generated.queries]

        found_urls = []
        for q in queries_to_run:
            if q not in state.visited_queries:
                state.visited_queries.append(q)
                items = self.search_tool.search(q, max_results=self.limits.max_search_results)
                for item in items:
                    if item.url and item.url not in found_urls:
                        found_urls.append(item.url)
                        # Register source stub
                        source_id = f"src_{uuid.uuid4().hex[:8]}"
                        if source_id not in state.source_ids and len(state.source_ids) < self.limits.max_sources_per_run:
                            state.source_ids.append(source_id)
                            domain = item.url.split("/")[2] if "://" in item.url else ""
                            self.source_repo.add(
                                source_id=source_id,
                                session_id=state.session_id,
                                url=item.url,
                                title=item.title,
                                domain=domain
                            )

        self.session_repo.update_phase_and_step(
            session_id=state.session_id,
            phase=state.phase.value,
            step=state.step,
            status=state.status.value
        )
        return found_urls

    def run_fetch_phase(self, state: ResearchState, urls: Optional[List[str]] = None) -> None:
        """Fetches web pages, cleans HTML via Trafilatura, and stores raw evidence baseline."""
        self.state_machine.transition(state, ResearchPhase.FETCH, reason="Fetching and caching source documents")
        sources = self.source_repo.get_by_session(state.session_id)
        
        target_urls = urls or [s["url"] for s in sources if s.get("url")]
        limit = min(len(target_urls), self.limits.max_fetch_calls)

        for url in target_urls[:limit]:
            fetched = self.fetch_tool.fetch(url)
            if fetched and fetched.text:
                logger.info(f"[ResearchEngine] Successfully parsed clean text ({len(fetched.text)} chars) from {url}")

        self.session_repo.update_phase_and_step(
            session_id=state.session_id,
            phase=state.phase.value,
            step=state.step,
            status=state.status.value
        )
