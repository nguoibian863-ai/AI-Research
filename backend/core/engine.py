import uuid
import json
import logging
from typing import Optional, List, Dict, Any, Union
from backend.core.state import (
    ResearchState,
    ResearchPhase,
    ResearchStatus,
    ResearchPlan,
    PlanTask,
    OpenQuestion
)
from backend.core.limits import ResearchLimits
from backend.core.budget import ExecutionBudgetTracker
from backend.core.transitions import StateMachine
from backend.core.errors import BudgetExceededError, StateTransitionError
from backend.llm.backend import LLMBackend
from backend.llm.schemas import ResearchPlanSchema, GeneratedQueriesSchema
from backend.db.database import DatabaseManager
from backend.db.repositories import (
    SessionRepository,
    QueryRepository,
    SourceRepository,
    DocumentRepository,
    RawEvidenceRepository,
    EvidenceRepository,
    ClaimRepository,
    ReportRepository,
    TrajectoryRepository
)
from backend.tools.web_search import WebSearchTool
from backend.tools.web_fetch import WebFetchTool, FetchedWebContent

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
        self.query_repo = QueryRepository(self.db)
        self.source_repo = SourceRepository(self.db)
        self.doc_repo = DocumentRepository(self.db)
        self.raw_evidence_repo = RawEvidenceRepository(self.db)
        self.evidence_repo = EvidenceRepository(self.db)
        self.claim_repo = ClaimRepository(self.db)
        self.report_repo = ReportRepository(self.db)
        self.trajectory_repo = TrajectoryRepository(self.db)

        # Session Budget Trackers
        self._budget_trackers: Dict[str, ExecutionBudgetTracker] = {}

    def get_budget_tracker(self, session_id: str) -> ExecutionBudgetTracker:
        if session_id not in self._budget_trackers:
            self._budget_trackers[session_id] = ExecutionBudgetTracker(limits=self.limits)
        return self._budget_trackers[session_id]

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
        self._budget_trackers[sid] = ExecutionBudgetTracker(limits=self.limits)
        logger.info(f"[{sid}][INIT] Created session with goal: '{goal}'")
        return state

    def load_state(self, session_id: str) -> ResearchState:
        """P0: Full state restoration directly from SQLite Source of Truth."""
        row = self.session_repo.get(session_id)
        if not row:
            raise KeyError(f"Research session '{session_id}' not found in database.")

        plan = None
        if row.get("plan_json"):
            try:
                plan = ResearchPlan.model_validate_json(row["plan_json"])
            except Exception as e:
                logger.warning(f"[{session_id}] Failed to deserialize plan_json: {e}")

        open_questions = []
        if row.get("open_questions_json"):
            try:
                raw_qs = json.loads(row["open_questions_json"])
                open_questions = [OpenQuestion.model_validate(q) for q in raw_qs]
            except Exception as e:
                logger.warning(f"[{session_id}] Failed to deserialize open_questions: {e}")

        visited_queries = self.query_repo.get_visited(session_id)
        source_ids = self.source_repo.get_ids_by_session(session_id)
        evidence_ids = self.evidence_repo.get_ids_by_session(session_id)
        claim_ids = self.claim_repo.get_ids_by_session(session_id)

        state = ResearchState(
            session_id=session_id,
            goal=row["goal"],
            plan=plan,
            visited_queries=visited_queries,
            source_ids=source_ids,
            evidence_ids=evidence_ids,
            claim_ids=claim_ids,
            open_questions=open_questions,
            step=row["step"],
            phase=ResearchPhase(row["phase"]),
            status=ResearchStatus(row["status"])
        )
        logger.info(
            f"[{session_id}][RESTORE] Full state restored: phase={state.phase.value}, step={state.step}, "
            f"has_plan={state.plan is not None}, queries={len(state.visited_queries)}, sources={len(state.source_ids)}"
        )
        return state

    def save_state(self, state: ResearchState) -> None:
        """Persists ResearchState updates to SQLite."""
        plan_json = state.plan.model_dump_json() if state.plan else None
        open_q_json = json.dumps([q.model_dump() for q in state.open_questions])
        self.session_repo.update_state(
            session_id=state.session_id,
            phase=state.phase.value,
            step=state.step,
            status=state.status.value,
            plan_json=plan_json,
            open_questions_json=open_q_json
        )

    def run_plan_phase(self, state: ResearchState) -> None:
        """Executes the PLAN phase with strict budget checks."""
        budget = self.get_budget_tracker(state.session_id)
        budget.assert_can_call_llm()

        self.state_machine.transition(state, ResearchPhase.PLAN, reason="Starting research plan")
        prompt = (
            f"You are a Senior Research Planner. Given the following research question, "
            f"break it down into 3-5 concrete, factual investigation sub-tasks and identify core hypotheses to test.\n\n"
            f"Question: {state.goal}"
        )
        res = self.llm.structured_generate(prompt, schema=ResearchPlanSchema)
        budget.record_llm_call(tokens=res.total_tokens)

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

        self.save_state(state)

        # Save trajectory sample in raw partition
        self.trajectory_repo.add(
            trajectory_id=f"traj_{uuid.uuid4().hex[:12]}",
            session_id=state.session_id,
            task_type="planning",
            model_source=getattr(self.llm, "model", "mock"),
            payload={"goal": state.goal, "plan": state.plan.model_dump()},
            partition="raw"
        )
        logger.info(f"[{state.session_id}][PLAN] Plan generated with {len(tasks)} tasks.")

    def run_search_phase(self, state: ResearchState, custom_queries: Optional[List[str]] = None) -> List[str]:
        """Generates targeted search queries and executes web search with URL deduplication & source limits."""
        budget = self.get_budget_tracker(state.session_id)
        self.state_machine.transition(state, ResearchPhase.SEARCH, reason="Executing web search")

        queries_to_run = custom_queries or []
        if not queries_to_run:
            budget.assert_can_call_llm()
            prompt = (
                f"Based on the research goal and plan, generate 2-4 search queries across categories "
                f"(discovery, evidence, verification, contradiction) to gather required facts.\n\n"
                f"Goal: {state.goal}\n"
                f"Plan: {state.plan.model_dump_json() if state.plan else ''}\n"
                f"Previously visited: {state.visited_queries}"
            )
            res = self.llm.structured_generate(prompt, schema=GeneratedQueriesSchema)
            budget.record_llm_call(tokens=res.total_tokens)
            generated: GeneratedQueriesSchema = res.parsed
            queries_to_run = [q.query for q in generated.queries]

        existing_urls = self.source_repo.get_urls_by_session(state.session_id)
        found_urls: List[str] = []

        for q in queries_to_run:
            if not budget.can_search():
                logger.warning(f"[{state.session_id}][SEARCH] Search budget exhausted ({budget.search_calls} calls). Skipping query: {q}")
                break

            if q not in state.visited_queries:
                state.visited_queries.append(q)
                self.query_repo.add(
                    query_id=f"qry_{uuid.uuid4().hex[:8]}",
                    session_id=state.session_id,
                    query_text=q,
                    query_type="discovery"
                )
                budget.record_search()

                items = self.search_tool.search(q, max_results=self.limits.max_search_results)
                for item in items:
                    # Enforce max sources limit
                    if len(state.source_ids) >= self.limits.max_sources_per_run:
                        logger.info(f"[{state.session_id}][SEARCH] Reached max_sources_per_run limit ({self.limits.max_sources_per_run}).")
                        break

                    url = item.url.strip() if item.url else ""
                    if url and url not in existing_urls and url not in found_urls:
                        found_urls.append(url)
                        existing_urls.add(url)
                        source_id = f"src_{uuid.uuid4().hex[:8]}"
                        state.source_ids.append(source_id)
                        domain = url.split("/")[2] if "://" in url else ""
                        self.source_repo.add(
                            source_id=source_id,
                            session_id=state.session_id,
                            url=url,
                            title=item.title,
                            domain=domain,
                            canonical_key=url.lower().rstrip("/")
                        )

        self.save_state(state)
        logger.info(f"[{state.session_id}][SEARCH] Completed. Found {len(found_urls)} new URLs, total sources: {len(state.source_ids)}.")
        return found_urls

    def run_fetch_phase(self, state: ResearchState, urls: Optional[List[str]] = None) -> List[FetchedWebContent]:
        """Fetches web pages, cleans HTML via Trafilatura, stores document and raw evidence baseline."""
        budget = self.get_budget_tracker(state.session_id)
        self.state_machine.transition(state, ResearchPhase.FETCH, reason="Fetching source documents")

        sources = self.source_repo.get_by_session(state.session_id)
        target_urls = urls or [s["url"] for s in sources if s.get("url")]

        fetched_documents: List[FetchedWebContent] = []

        for url in target_urls:
            if not budget.can_fetch():
                logger.warning(f"[{state.session_id}][FETCH] Fetch budget exhausted ({budget.fetch_calls} calls). Skipping: {url}")
                break

            budget.record_fetch()
            fetched = self.fetch_tool.fetch(url)
            if fetched and fetched.text:
                fetched_documents.append(fetched)
                # Find matching source
                matching_source = next((s for s in sources if s["url"] == url), None)
                source_id = matching_source["source_id"] if matching_source else state.source_ids[0]

                # Save document
                doc_id = f"doc_{uuid.uuid4().hex[:8]}"
                self.doc_repo.add(
                    doc_id=doc_id,
                    source_id=source_id,
                    file_path=None,
                    content_hash=fetched.content_hash,
                    raw_text=fetched.text
                )

                # Register raw evidence baseline
                raw_ev_id = f"raw_{uuid.uuid4().hex[:8]}"
                sample_quote = fetched.text[:300].strip()
                self.raw_evidence_repo.add(
                    raw_evidence_id=raw_ev_id,
                    session_id=state.session_id,
                    source_id=source_id,
                    raw_quote=sample_quote,
                    char_start=0,
                    char_end=len(sample_quote)
                )

        self.save_state(state)
        logger.info(f"[{state.session_id}][FETCH] Fetched {len(fetched_documents)} documents successfully.")
        return fetched_documents

    def run_basic_answer(self, state: ResearchState, fetched_docs: List[FetchedWebContent]) -> str:
        """Week 1 Deliverable: Synthesizes a grounded initial answer from fetched sources."""
        budget = self.get_budget_tracker(state.session_id)
        budget.assert_can_call_llm()

        # Build context from fetched documents
        context_snippets = []
        for i, doc in enumerate(fetched_docs[:5], start=1):
            snippet = doc.text[:1200]
            context_snippets.append(f"--- Source [{i}] ({doc.url}) ---\n{snippet}\n")
        full_context = "\n".join(context_snippets) if context_snippets else "No external documents retrieved."

        prompt = (
            f"You are a Research Analyst. Provide a clear, factual, and grounded answer to the question "
            f"based strictly on the gathered context.\n\n"
            f"Question: {state.goal}\n\n"
            f"Gathered Evidence Context:\n{full_context}\n\n"
            f"Synthesize the key findings, metrics, and comparisons directly addressing the question."
        )

        # Transition to WRITE phase
        self.state_machine.transition(state, ResearchPhase.WRITE, reason="Synthesizing report")
        res = self.llm.generate(prompt)
        budget.record_llm_call(tokens=res.total_tokens)
        answer = res.content.strip()

        # Transition state to DONE (or PARTIAL if no docs)
        target_phase = ResearchPhase.DONE if fetched_docs else ResearchPhase.PARTIAL
        self.state_machine.transition(state, target_phase, reason="Report synthesis completed")
        self.save_state(state)

        # Save report
        report_id = f"rep_{uuid.uuid4().hex[:8]}"
        self.report_repo.create(
            report_id=report_id,
            session_id=state.session_id,
            title=f"Research: {state.goal[:60]}",
            content_markdown=answer,
            status=state.status.value
        )

        # Save final trajectory
        self.trajectory_repo.add(
            trajectory_id=f"traj_{uuid.uuid4().hex[:12]}",
            session_id=state.session_id,
            task_type="answer_synthesis",
            model_source=getattr(self.llm, "model", "mock"),
            payload={"goal": state.goal, "answer": answer, "sources_count": len(fetched_docs)},
            verified=True if fetched_docs else False,
            quality_score=0.90 if fetched_docs else 0.50,
            partition="candidate"
        )

        logger.info(f"[{state.session_id}][ANSWER] Final report saved (report_id={report_id}, status={state.status.value}).")
        return answer

    def run_week1(self, session_or_goal: Union[ResearchState, str]) -> Dict[str, Any]:
        """
        Week 1 End-to-End Orchestrator:
        Question -> PLAN -> SEARCH -> FETCH -> ANSWER -> DONE
        """
        if isinstance(session_or_goal, str):
            # Check if existing session_id or new goal
            if session_or_goal.startswith("sess_") and self.session_repo.get(session_or_goal):
                state = self.load_state(session_or_goal)
            else:
                state = self.create_session(goal=session_or_goal)
        else:
            state = session_or_goal

        budget = self.get_budget_tracker(state.session_id)
        logger.info(f"[{state.session_id}][E2E] Starting end-to-end research for: '{state.goal}'")

        # 1. Plan
        self.run_plan_phase(state)

        # 2. Search
        urls = self.run_search_phase(state)

        # 3. Fetch
        docs = self.run_fetch_phase(state, urls=urls)

        # 4. Answer
        answer = self.run_basic_answer(state, fetched_docs=docs)

        return {
            "session_id": state.session_id,
            "goal": state.goal,
            "phase": state.phase.value,
            "status": state.status.value,
            "plan": state.plan.model_dump() if state.plan else None,
            "visited_queries": state.visited_queries,
            "sources_count": len(state.source_ids),
            "documents_fetched": len(docs),
            "answer": answer,
            "budget": budget.summary()
        }
