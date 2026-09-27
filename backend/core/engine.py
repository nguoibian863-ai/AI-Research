import uuid
import json
import logging
import re
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
from backend.core.errors import (
    BudgetExceededError,
    StateTransitionError,
    ResearchException,
    SessionNotFoundError
)
from backend.llm.backend import LLMBackend
from backend.llm.schemas import ResearchPlanSchema, GeneratedQueriesSchema
from backend.db.database import DatabaseManager
from backend.db.repositories import (
    SessionRepository,
    QueryRepository,
    SourceRepository,
    DocumentRepository,
    ChunkRepository,
    RawEvidenceRepository,
    EvidenceRepository,
    ClaimRepository,
    ReportRepository,
    TrajectoryRepository
)
from backend.tools.web_search import WebSearchTool
from backend.tools.web_fetch import WebFetchTool, FetchedWebContent
from backend.tools.pdf_fetch import PDFFetchTool, FetchedPDFContent
from backend.parsing.html_cleaner import HTMLCleaner, CleanedDocument
from backend.parsing.chunker import SectionAwareChunker, ParsedChunk
from backend.retrieval.bm25 import BM25Index
from backend.retrieval.embeddings import EmbeddingBackend, LocalHashEmbeddingBackend
from backend.retrieval.faiss_index import FaissVectorIndex
from backend.retrieval.hybrid import HybridRetriever, RetrievedChunk
from backend.core.coverage import (
    extract_core_entities,
    check_entity_in_text,
    evaluate_coverage,
    evaluate_evidence_coverage,
    GENERIC_RESEARCH_TERMS,
    DATASET_BENCHMARK_TERMS,
    extract_context_entities,
    text_mentions_entity,
    extract_substantive_numbers,
    subject_matches_entity
)
from backend.evidence.extractor import EvidenceExtractor
from backend.evidence.verifier import CitationVerifier
from backend.sources.dedup import compute_canonical_key
from backend.sources.credibility import score_source_credibility
from backend.verification.nli import (
    BaseNLIVerifier,
    CompositeNLIVerifier,
    ClaimVerificationPipeline,
    NLI_PROMPT_VERSION,
    ALL_KNOWN_METRICS,
    HARDWARE_ENV_TERMS,
    STOPWORDS,
    LINKING_VERBS,
    GENERIC_DOMAIN_NOUNS,
    ALL_COMPARATIVES,
    CONTEXT_EXCLUDED,
    strip_citation_markers,
)
from backend.verification.schemas import NLILabel

logger = logging.getLogger(__name__)

BLOCKED_DOMAINS = {
    "facebook.com", "instagram.com", "twitter.com", "x.com", "pinterest.com",
    "tiktok.com", "quora.com", "youtube.com"
}



class ResearchEngine:
    def __init__(
        self,
        llm: LLMBackend,
        db: DatabaseManager,
        limits: Optional[ResearchLimits] = None,
        search_tool: Optional[WebSearchTool] = None,
        fetch_tool: Optional[WebFetchTool] = None,
        pdf_tool: Optional[PDFFetchTool] = None,
        embedding_backend: Optional[EmbeddingBackend] = None,
        chunker: Optional[SectionAwareChunker] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        nli_verifier: Optional[BaseNLIVerifier] = None
    ):
        self._llm = llm
        self.db = db
        self.limits = limits or ResearchLimits()
        self.state_machine = StateMachine(limits=self.limits)
        self.search_tool = search_tool or WebSearchTool(max_results=self.limits.max_search_results)
        self.fetch_tool = fetch_tool or WebFetchTool()
        self.pdf_tool = pdf_tool or PDFFetchTool()
        self.chunker = chunker or SectionAwareChunker(
            max_chunk_tokens=self.limits.max_chunk_tokens,
            max_chunk_chars=self.limits.max_chunk_chars
        )
        self.embedding_backend = embedding_backend or LocalHashEmbeddingBackend()
        
        # Per-session hybrid retrievers (BM25 + FAISS isolation)

        self._session_retrievers: Dict[str, HybridRetriever] = {}
        # chunk_ids currently embedded in each session's index (avoids re-embedding on every query)
        self._indexed_chunk_ids: Dict[str, List[str]] = {}
        if hybrid_retriever:
            self._session_retrievers["default"] = hybrid_retriever

        # Repositories
        self.session_repo = SessionRepository(self.db)
        self.query_repo = QueryRepository(self.db)
        self.source_repo = SourceRepository(self.db)
        self.doc_repo = DocumentRepository(self.db)
        self.chunk_repo = ChunkRepository(self.db)
        self.raw_evidence_repo = RawEvidenceRepository(self.db)
        self.evidence_repo = EvidenceRepository(self.db)
        self.claim_repo = ClaimRepository(self.db)
        self.report_repo = ReportRepository(self.db)
        self.trajectory_repo = TrajectoryRepository(self.db)
        self.evidence_extractor = EvidenceExtractor(llm=self.llm, db=self.db)
        self.citation_verifier = CitationVerifier()
        self.nli_verifier = nli_verifier or CompositeNLIVerifier(llm=self.llm)
        self.claim_verification_pipeline = ClaimVerificationPipeline(verifier=self.nli_verifier)

        # Session Budget Trackers (cached in-memory, backed by SQLite)
        self._budget_trackers: Dict[str, ExecutionBudgetTracker] = {}

    @property
    def llm(self) -> Optional[LLMBackend]:
        return self._llm

    @llm.setter
    def llm(self, val: Optional[LLMBackend]) -> None:
        self._llm = val
        if hasattr(self, "evidence_extractor") and self.evidence_extractor is not None:
            self.evidence_extractor.llm = val

    def _get_model_source(self) -> str:
        model = getattr(self.llm, "model", None)
        if isinstance(model, str) and model:
            return model
        return "mock"

    @staticmethod
    def _goal_subject_entities(goal: str) -> List[str]:
        """Core entities of the goal that are compared subjects (datasets/benchmarks/context excluded)."""
        context = set(DATASET_BENCHMARK_TERMS) | extract_context_entities(goal)
        subjects = [e for e in extract_core_entities(goal) if e not in context]
        return subjects or extract_core_entities(goal)

    @classmethod
    def _is_claim_on_topic(cls, sent: str, cited_evs: List[Dict[str, Any]], goal_core_entities: Optional[List[str]]) -> bool:
        """
        Determines whether a cited claim is on-topic relative to the research goal entities.
        Rules:
        1. If the claim text directly mentions any goal entity -> ON_TOPIC.
        2. If the claim explicitly mentions OTHER model/architecture names (e.g. YOLOv10, YOLOv6) without naming any goal entity -> strictly OFF_TOPIC (even if cited evidence has goal subject!).
        3. If the claim contains NO other model names (e.g. pronoun 'It achieves...', 'This model reaches...') -> check if cited evidence subject matches a goal entity.
        """
        if not goal_core_entities:
            return True

        # 1. Direct mention in claim text
        has_goal_entity = any(
            check_entity_in_text(ent, sent) or subject_matches_entity(sent, ent)
            for ent in goal_core_entities
        )
        if has_goal_entity:
            return True

        # 2. Extract model/architecture names from claim
        sent_clean = strip_citation_markers(sent)
        non_model = (
            DATASET_BENCHMARK_TERMS
            | HARDWARE_ENV_TERMS
            | ALL_KNOWN_METRICS
            | STOPWORDS
            | LINKING_VERBS
            | GENERIC_DOMAIN_NOUNS
            | ALL_COMPARATIVES
            | CONTEXT_EXCLUDED
            | GENERIC_RESEARCH_TERMS
            | {
                "according", "val", "train", "test", "dataset", "datasets",
                "benchmark", "benchmarks", "paper", "authors", "result", "results"
            }
        )
        claim_models = [
            e for e in extract_core_entities(sent_clean)
            if e not in non_model and len(e) > 1 and not e.isdigit() and not re.match(r"^e\d+$", e)
        ]

        # If claim explicitly names other models without naming any goal entity -> OFF_TOPIC
        if claim_models:
            return False

        # 3. Only for pronoun / generic references: check cited evidence subjects
        for ev in cited_evs:
            ev_subj = ev.get("subject") or ""
            if ev_subj and any(
                check_entity_in_text(ent, ev_subj) or subject_matches_entity(ev_subj, ent)
                for ent in goal_core_entities
            ):
                return True

        return False

    def get_session_retriever(self, session_id: str) -> HybridRetriever:
        """Returns or creates an isolated HybridRetriever per session."""
        if session_id not in self._session_retrievers:
            self._session_retrievers[session_id] = HybridRetriever(
                bm25_index=BM25Index(),
                vector_index=FaissVectorIndex(embedding_backend=self.embedding_backend)
            )
        return self._session_retrievers[session_id]

    @property
    def hybrid_retriever(self) -> HybridRetriever:
        return self.get_session_retriever("default")

    @hybrid_retriever.setter
    def hybrid_retriever(self, val: HybridRetriever) -> None:
        self._session_retrievers["default"] = val

    def _sync_retriever_index(self, session_id: str, retriever: HybridRetriever, chunks: List[Dict[str, Any]]) -> None:
        """
        Keeps the session index in sync with SQLite without re-embedding unchanged chunks.
        Same chunk set -> no-op; new chunks appended -> embed only the new ones; otherwise full rebuild.
        """
        current_ids = [c["chunk_id"] for c in chunks]
        indexed_ids = self._indexed_chunk_ids.get(session_id)

        if indexed_ids == current_ids:
            return

        if indexed_ids is not None and set(indexed_ids).issubset(current_ids):
            indexed = set(indexed_ids)
            new_chunks = [c for c in chunks if c["chunk_id"] not in indexed]
            retriever.bm25_index.index(chunks)  # cheap, keeps IDF consistent over the full corpus
            retriever.vector_index.add_chunks(new_chunks)
            self._indexed_chunk_ids[session_id] = list(indexed_ids) + [c["chunk_id"] for c in new_chunks]
            logger.info(f"[{session_id}][RETRIEVE] Index updated incrementally: +{len(new_chunks)} chunks.")
            return

        retriever.index_chunks(chunks)
        self._indexed_chunk_ids[session_id] = current_ids

    def retrieve_chunks(self, session_id: str, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Read-only chunk retrieval without mutating session state or phase."""
        chunks = self.chunk_repo.get_by_session(session_id)
        if not chunks:
            return []
        retriever = self.get_session_retriever(session_id)
        self._sync_retriever_index(session_id, retriever, chunks)
        k = top_k or self.limits.retrieval_top_k
        return retriever.retrieve(query=query, top_k=k)


    def get_budget_tracker(self, session_id: str) -> ExecutionBudgetTracker:
        if session_id not in self._budget_trackers:
            row = self.session_repo.get(session_id)
            if row:
                self._budget_trackers[session_id] = ExecutionBudgetTracker(
                    limits=self.limits,
                    search_calls=row.get("search_calls") or 0,
                    fetch_calls=row.get("fetch_calls") or 0,
                    llm_calls=row.get("llm_calls") or 0,
                    tokens_consumed=row.get("tokens_consumed") or 0,
                    step_count=row.get("step") or 0
                )
            else:
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
            raise SessionNotFoundError(f"Research session '{session_id}' not found in database.")

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

        # Restore budget tracker with persistent counters
        self._budget_trackers[session_id] = ExecutionBudgetTracker(
            limits=self.limits,
            search_calls=row.get("search_calls") or 0,
            fetch_calls=row.get("fetch_calls") or 0,
            llm_calls=row.get("llm_calls") or 0,
            tokens_consumed=row.get("tokens_consumed") or 0,
            step_count=row.get("step") or 0
        )

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
            status=ResearchStatus(row["status"]),
            error_message=row.get("error_message")
        )
        logger.info(
            f"[{session_id}][RESTORE] Full state restored: phase={state.phase.value}, step={state.step}, "
            f"has_plan={state.plan is not None}, queries={len(state.visited_queries)}, sources={len(state.source_ids)}"
        )
        return state

    def save_state(self, state: ResearchState) -> None:
        """Persists ResearchState and cumulative budget counters to SQLite."""
        plan_json = state.plan.model_dump_json() if state.plan else None
        open_q_json = json.dumps([q.model_dump() for q in state.open_questions])
        budget = self._budget_trackers.get(state.session_id)

        self.session_repo.update_state(
            session_id=state.session_id,
            phase=state.phase.value,
            step=state.step,
            status=state.status.value,
            plan_json=plan_json,
            open_questions_json=open_q_json,
            error_message=state.error_message,
            search_calls=budget.search_calls if budget else None,
            fetch_calls=budget.fetch_calls if budget else None,
            llm_calls=budget.llm_calls if budget else None,
            tokens_consumed=budget.total_tokens_consumed if budget else None
        )

    def _handle_phase_error(self, state: ResearchState, error: Exception, phase_name: str) -> None:
        """
        Handles runtime failures while strictly preserving state machine invariants.
        CRITICAL RULES:
        1. Terminal states are immutable: If phase is DONE, PARTIAL, FAILED, or CANCELLED, do not touch state.phase!
        2. Validation / Transition errors (e.g. StateTransitionError): The caller sent an invalid request.
           Do NOT corrupt the session state in the database!
        3. Runtime errors (e.g. ModelInferenceError, IO error): Transition cleanly via StateMachine
           to FAILED or PARTIAL (if sources were collected), record error_message, and save state.
           NEVER bypass the state machine with direct assignment!
        """
        logger.error(f"[{state.session_id}][{phase_name}] Failure: {error}")

        # Rule 1 & 2: Do not alter state for client validation errors or if already terminal
        if isinstance(error, StateTransitionError):
            return

        if state.phase in {ResearchPhase.DONE, ResearchPhase.PARTIAL, ResearchPhase.FAILED, ResearchPhase.CANCELLED}:
            return

        state.error_message = str(error)
        target_phase = ResearchPhase.PARTIAL if state.source_ids else ResearchPhase.FAILED
        try:
            self.state_machine.transition(state, target_phase, reason=f"Runtime error in {phase_name}: {error}")
            self.save_state(state)
        except Exception as transition_err:
            logger.warning(f"[{state.session_id}] Could not transition to {target_phase}: {transition_err}")
            self.save_state(state)

    def run_plan_phase(self, state: ResearchState) -> None:
        """Executes the PLAN phase with strict budget checks and technical few-shot prompt."""
        try:
            budget = self.get_budget_tracker(state.session_id)
            budget.assert_can_call_llm()

            self.state_machine.transition(state, ResearchPhase.PLAN, reason="Starting research plan")
            prompt = (
                f"You are a Senior Technical Research Planner. Given the following research question, "
                f"break it down into 3-5 concrete, factual investigation sub-tasks focusing on measurable benchmarks, "
                f"hardware specifications, datasets, and baseline comparisons.\n"
                f"IMPORTANT: You are planning a LITERATURE search. Do NOT plan to run experiments, train models, "
                f"collect or preprocess datasets, or tune configurations yourself. Every task must FIND already "
                f"published results (papers, official benchmarks, documentation) for the entities in the question.\n\n"
                f"Example Benchmark Plan:\n"
                f"Goal: Compare RocksDB vs LevelDB write amplification and throughput on NVMe SSDs\n"
                f"Tasks:\n"
                f"1. Investigate RocksDB LSM-tree compaction strategies, write stall mitigation, and random write throughput.\n"
                f"2. Investigate LevelDB write amplification factor, memory table architecture, and sequential read/write benchmarks.\n"
                f"3. Compare CPU overhead, space amplification, and P99 latency on high-iops NVMe storage.\n\n"
                f"Question: {state.goal}"
            )
            res = self.llm.structured_generate(prompt, schema=ResearchPlanSchema, num_predict=512)
            budget.record_llm_call(tokens=res.total_tokens, count=getattr(res, "calls_made", 1))

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
                model_source=self._get_model_source(),
                payload={"goal": state.goal, "plan": state.plan.model_dump()},
                partition="raw"
            )
            logger.info(f"[{state.session_id}][PLAN] Plan generated with {len(tasks)} tasks.")
        except Exception as e:
            self._handle_phase_error(state, e, "PLAN")
            raise

    def run_search_phase(self, state: ResearchState, custom_queries: Optional[List[str]] = None) -> List[str]:
        """Generates targeted search queries and executes web search with URL deduplication & domain filtering."""
        try:
            budget = self.get_budget_tracker(state.session_id)
            self.state_machine.transition(state, ResearchPhase.SEARCH, reason="Executing web search")

            queries_to_run = custom_queries or []
            query_origin = "open_questions" if custom_queries else "llm"
            if not queries_to_run:
                budget.assert_can_call_llm()
                prompt = (
                    f"Based on the research goal and plan, generate 2-4 concise, highly-targeted keyword search queries "
                    f"(3 to 8 words per query). Do NOT use conversational question sentences.\n"
                    f"Each query must name at least one entity from the goal and target published results "
                    f"(benchmark numbers, papers, official docs). Do NOT search for configuration parameters, "
                    f"installation steps or tooling.\n\n"
                    f"Good Examples:\n"
                    f"- 'RocksDB LevelDB write amplification benchmark NVMe'\n"
                    f"- 'RocksDB compaction throughput latency SSD'\n"
                    f"- 'LevelDB LSM tree space amplification evaluation'\n\n"
                    f"Goal: {state.goal}\n"
                    f"Plan: {state.plan.model_dump_json() if state.plan else ''}\n"
                    f"Previously visited: {state.visited_queries}"
                )
                res = self.llm.structured_generate(prompt, schema=GeneratedQueriesSchema, num_predict=256)
                budget.record_llm_call(tokens=res.total_tokens, count=getattr(res, "calls_made", 1))
                generated: GeneratedQueriesSchema = res.parsed
                queries_to_run = [q.query for q in generated.queries]

            existing_urls = self.source_repo.get_urls_by_session(state.session_id)
            found_urls: List[str] = []
            goal_entities = self._goal_subject_entities(state.goal)

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
                        if len(state.source_ids) >= self.limits.max_sources_per_run:
                            logger.info(f"[{state.session_id}][SEARCH] Reached max_sources_per_run limit ({self.limits.max_sources_per_run}).")
                            break

                        url = item.url.strip() if item.url else ""
                        if not url:
                            continue

                        raw_domain = url.split("/")[2].lower() if "://" in url else ""
                        domain = raw_domain.split(":")[0]
                        if any(domain == b or domain.endswith("." + b) for b in BLOCKED_DOMAINS):
                            logger.info(f"[{state.session_id}][SEARCH] Skipping blocked domain: {domain}")
                            continue

                        # Relevance filter: whole-word matching against core entities of query
                        query_entities = extract_core_entities(q)
                        item_content = f"{item.title} {item.snippet} {url}"
                        if query_entities and not any(check_entity_in_text(ent, item_content) for ent in query_entities):
                            logger.info(f"[{state.session_id}][SEARCH] Skipping irrelevant search result (no core entity match): '{item.title}'")
                            continue

                        canonical_k = compute_canonical_key(url, item.title)
                        existing_source = self.source_repo.find_by_canonical_key(state.session_id, canonical_k)
                        if existing_source:
                            logger.info(f"[{state.session_id}][SEARCH] Deduplicated source by canonical key '{canonical_k}': {url}")
                            continue

                        if url not in existing_urls and url not in found_urls:
                            # Plan 14.2: score source before fetching; low-credibility sources are skipped
                            cred = score_source_credibility(
                                url=url,
                                title=item.title,
                                domain=domain,
                                text=item.snippet,
                                published_at=getattr(item, "published_at", None),
                                goal_entities=goal_entities
                            )
                            if cred["score"] < self.limits.min_source_score:
                                logger.info(
                                    f"[{state.session_id}][SEARCH] Skipping low-credibility source "
                                    f"(score {cred['score']} < {self.limits.min_source_score}, {cred['tier']}): {url}"
                                )
                                continue

                            found_urls.append(url)
                            existing_urls.add(url)
                            source_id = f"src_{uuid.uuid4().hex[:8]}"
                            state.source_ids.append(source_id)
                            self.source_repo.add(
                                source_id=source_id,
                                session_id=state.session_id,
                                url=url,
                                title=item.title,
                                source_type=getattr(item, "source_type", "web"),
                                domain=domain,
                                canonical_key=canonical_k,
                                published_at=getattr(item, "published_at", None),
                                credibility_score=cred["score"],
                                credibility_details=cred
                            )

            self.save_state(state)

            # Plan 43.1: Log query_generation trajectory
            has_sources = len(found_urls) > 0
            self.trajectory_repo.add(
                trajectory_id=f"traj_{uuid.uuid4().hex[:12]}",
                session_id=state.session_id,
                task_type="query_generation",
                model_source=self._get_model_source(),
                payload={
                    "goal": state.goal,
                    "plan": state.plan.model_dump() if state.plan else None,
                    "queries": queries_to_run,
                    "query_origin": query_origin,
                    "found_urls": found_urls,
                    "relevant_sources_count": len(found_urls),
                    "total_sources": len(state.source_ids),
                    "metadata": {
                        "task_type": "query_generation",
                        "model_source": self._get_model_source(),
                        "query_origin": query_origin,
                        "verified": False,
                        "quality_score": min(1.0, len(found_urls) / max(1, len(queries_to_run))),
                        "source_ids": list(state.source_ids),
                        "language": "en"
                    }
                },
                verified=False,
                quality_score=min(1.0, len(found_urls) / max(1, len(queries_to_run))),
                partition="raw"
            )
            logger.info(f"[{state.session_id}][SEARCH] Completed. Found {len(found_urls)} new URLs, total sources: {len(state.source_ids)}.")
            return found_urls
        except Exception as e:
            self._handle_phase_error(state, e, "SEARCH")
            raise

    def run_fetch_phase(self, state: ResearchState, urls: Optional[List[str]] = None) -> List[Union[FetchedWebContent, FetchedPDFContent]]:
        """
        Fetches web pages or PDFs, parses them into CleanedDocuments, and stores chunks in SQLite.
        Routes PDF URLs to PDFFetchTool, captures author/date metadata, and respects max_total_chunks.
        """
        try:
            budget = self.get_budget_tracker(state.session_id)
            self.state_machine.transition(state, ResearchPhase.FETCH, reason="Fetching source documents")

            if urls is not None:
                target_urls = urls
            else:
                sources = self.source_repo.get_by_session(state.session_id)
                target_urls = [s["url"] for s in sources if s.get("url")]

            fetched_documents: List[Union[FetchedWebContent, FetchedPDFContent]] = []
            sources = self.source_repo.get_by_session(state.session_id)
            current_session_chunks = len(self.chunk_repo.get_by_session(state.session_id))

            for url in target_urls:
                if not budget.can_fetch():
                    logger.warning(f"[{state.session_id}][FETCH] Fetch budget exhausted ({budget.fetch_calls} calls). Skipping: {url}")
                    break

                if current_session_chunks >= self.limits.max_total_chunks:
                    logger.info(f"[{state.session_id}][FETCH] Reached max_total_chunks limit ({self.limits.max_total_chunks}).")
                    break

                budget.record_fetch()
                is_pdf_url = url.lower().endswith(".pdf") or "/pdf/" in url.lower()
                if is_pdf_url:
                    fetched = self.pdf_tool.fetch(url)
                else:
                    fetched = self.fetch_tool.fetch(url)
                    if fetched and getattr(fetched, "is_pdf", False):
                        fetched = self.pdf_tool.fetch(url)
                        is_pdf_url = True
                    elif (not fetched or not fetched.text) and "arxiv.org/html/" in url:
                        # Fallback arXiv /html/ to /pdf/ if /html/ gives 404/empty (Issue 1)
                        arxiv_m = re.search(r"arxiv\.org/html/(\d{4}\.\d{4,5}(?:v\d+)?)", url)
                        if arxiv_m:
                            pdf_url = f"https://arxiv.org/pdf/{arxiv_m.group(1)}.pdf"
                            logger.info(f"[{state.session_id}][FETCH] arXiv HTML unavailable for {url}, falling back to PDF: {pdf_url}")
                            fetched = self.pdf_tool.fetch(pdf_url)
                            if fetched and fetched.text:
                                is_pdf_url = True

                if fetched and fetched.text:
                    fetched_documents.append(fetched)

                    matching_source = next((s for s in sources if s["url"] == url), None)
                    authors_list = [fetched.author] if getattr(fetched, "author", None) else None
                    pub_date = getattr(fetched, "date", None)
                    if not matching_source:
                        domain = url.split("/")[2] if "://" in url else ""
                        source_id = f"src_{uuid.uuid4().hex[:8]}"
                        self.source_repo.add(
                            source_id=source_id,
                            session_id=state.session_id,
                            url=url,
                            title=fetched.title or url,
                            domain=domain,
                            canonical_key=compute_canonical_key(url, fetched.title or url),
                            authors=authors_list,
                            published_at=pub_date,
                            source_type="pdf" if is_pdf_url else "web",
                            goal_entities=self._goal_subject_entities(state.goal)
                        )
                        if source_id not in state.source_ids:
                            state.source_ids.append(source_id)
                    else:
                        source_id = matching_source["source_id"]

                    # Save document
                    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
                    file_path = getattr(fetched, "file_path", None)
                    self.doc_repo.add(
                        doc_id=doc_id,
                        source_id=source_id,
                        file_path=file_path,
                        content_hash=fetched.content_hash,
                        raw_text=fetched.text
                    )

                    # Contextual section-aware chunking
                    cleaned_doc = CleanedDocument(
                        title=fetched.title,
                        author=getattr(fetched, "author", None),
                        date=getattr(fetched, "date", None),
                        text=fetched.text,
                        sections=getattr(fetched, "sections", [])
                    )
                    chunks = self.chunker.chunk_document(doc_id=doc_id, document=cleaned_doc)
                    if chunks:
                        available_slots = max(0, self.limits.max_total_chunks - current_session_chunks)
                        chunks_to_add = chunks[:available_slots]
                        chunk_dicts = [
                            {
                                "chunk_id": c.chunk_id,
                                "doc_id": c.doc_id,
                                "text": c.text,
                                "page": c.page,
                                "section": c.section,
                                "char_start": c.char_start,
                                "char_end": c.char_end,
                                "token_count": c.token_count
                            }
                            for c in chunks_to_add
                        ]
                        self.chunk_repo.add_batch(chunk_dicts)
                        current_session_chunks += len(chunks_to_add)

            self.save_state(state)
            logger.info(f"[{state.session_id}][FETCH] Fetched {len(fetched_documents)} documents successfully.")
            return fetched_documents
        except Exception as e:
            self._handle_phase_error(state, e, "FETCH")
            raise

    def run_clean_phase(self, state: ResearchState) -> int:
        """
        Executes the CLEAN phase: ensures all documents for the session are cleaned,
        section-aware chunked, and stored in SQLite chunks table up to max_total_chunks.
        """
        try:
            self.state_machine.transition(state, ResearchPhase.CLEAN, reason="Cleaning and chunking documents")
            sources = self.source_repo.get_by_session(state.session_id)
            total_chunks = len(self.chunk_repo.get_by_session(state.session_id))
            for s in sources:
                doc = self.doc_repo.get_by_source(s["source_id"])
                if doc:
                    existing_chunks = self.chunk_repo.get_by_doc(doc["doc_id"])
                    if not existing_chunks:
                        cleaned = None
                        if s.get("source_type") == "pdf" or (doc.get("file_path") and str(doc["file_path"]).lower().endswith(".pdf")):
                            try:
                                from backend.parsing.pdf_parser import PDFParser
                                cleaned = PDFParser().parse_file(doc["file_path"])
                            except Exception as pdf_err:
                                logger.warning(f"[{state.session_id}][CLEAN] PDF parsing failed ({pdf_err}), falling back to text.")
                        if cleaned is None:
                            try:
                                cleaner = HTMLCleaner()
                                cleaned = cleaner.clean(doc.get("raw_text") or "", url=s.get("url"))
                                if s.get("title") and not cleaned.title:
                                    cleaned.title = s.get("title")
                            except Exception as html_err:
                                logger.warning(f"[{state.session_id}][CLEAN] HTML cleaning failed ({html_err}), falling back to raw text.")
                                cleaned = CleanedDocument(
                                    title=s.get("title"),
                                    text=doc.get("raw_text") or ""
                                )

                        chunks = self.chunker.chunk_document(doc_id=doc["doc_id"], document=cleaned)
                        if chunks:
                            available_slots = max(0, self.limits.max_total_chunks - total_chunks)
                            chunks_to_add = chunks[:available_slots]
                            chunk_dicts = [
                                {
                                    "chunk_id": c.chunk_id,
                                    "doc_id": c.doc_id,
                                    "text": c.text,
                                    "page": c.page,
                                    "section": c.section,
                                    "char_start": c.char_start,
                                    "char_end": c.char_end,
                                    "token_count": c.token_count
                                }
                                for c in chunks_to_add
                            ]
                            self.chunk_repo.add_batch(chunk_dicts)
                            total_chunks += len(chunks_to_add)

            self.save_state(state)
            logger.info(f"[{state.session_id}][CLEAN] Documents chunked. Total chunks in SQLite: {total_chunks}")
            return total_chunks
        except Exception as e:
            self._handle_phase_error(state, e, "CLEAN")
            raise

    def run_retrieve_phase(self, state: ResearchState, query: Optional[str] = None, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Executes the RETRIEVE phase: builds hybrid BM25 + FAISS index from SQLite chunks
        and retrieves top-K chunks via Reciprocal Rank Fusion (RRF).
        Performs entity-balanced multi-query retrieval so distinct entities are not crowded out.
        """
        try:
            self.state_machine.transition(state, ResearchPhase.RETRIEVE, reason="Retrieving relevant evidence chunks")
            target_query = query or state.goal
            k = top_k or self.limits.retrieval_top_k

            # 1. Main query retrieval
            primary_results = self.retrieve_chunks(session_id=state.session_id, query=target_query, top_k=k)
            seen_ids = {c.chunk_id for c in primary_results}
            combined_results = list(primary_results)

            # 2. Entity-level queries for balanced coverage across distinct subjects (datasets excluded)
            core_entities = self._goal_subject_entities(state.goal)
            if len(core_entities) > 1:
                per_entity_k = max(2, (k + len(core_entities) - 1) // len(core_entities))
                for ent in core_entities:
                    ent_query = f"{ent} {target_query}"
                    ent_results = self.retrieve_chunks(session_id=state.session_id, query=ent_query, top_k=per_entity_k)
                    for c in ent_results:
                        if c.chunk_id not in seen_ids:
                            seen_ids.add(c.chunk_id)
                            combined_results.append(c)

            self.save_state(state)
            logger.info(f"[{state.session_id}][RETRIEVE] Retrieved {len(combined_results)} balanced chunks for query: '{target_query}'.")
            return combined_results
        except Exception as e:
            self._handle_phase_error(state, e, "RETRIEVE")
            raise

    def _normalize_chunk_for_evidence(self, c: Any) -> Dict[str, Any]:
        """Ensures chunk representation preserves source_id, url, and source_title for relational provenance."""
        if isinstance(c, dict):
            chunk_id = c.get("chunk_id", "")
            text = c.get("text", "")
            section = c.get("section", "")
            page = c.get("page")
            char_start = c.get("char_start", 0)
            char_end = c.get("char_end", 0)
            meta = c.get("metadata", {}) if isinstance(c.get("metadata"), dict) else {}
            source_id = c.get("source_id") or meta.get("source_id")
            url = c.get("url") or meta.get("url") or "local"
            source_title = c.get("source_title") or c.get("title") or meta.get("source_title") or meta.get("title") or ""
            doc_id = c.get("doc_id", "")
        else:
            chunk_id = getattr(c, "chunk_id", "")
            text = getattr(c, "text", "")
            section = getattr(c, "section", "")
            page = getattr(c, "page", None)
            char_start = getattr(c, "char_start", 0)
            char_end = getattr(c, "char_end", 0)
            meta = getattr(c, "metadata", {}) if hasattr(c, "metadata") and isinstance(c.metadata, dict) else {}
            source_id = getattr(c, "source_id", None) or meta.get("source_id")
            url = meta.get("url") or getattr(c, "url", "local")
            source_title = meta.get("source_title") or meta.get("title") or getattr(c, "source_title", "")
            doc_id = getattr(c, "doc_id", "")

        # Look up in SQLite if source_id is missing or placeholder
        if (not source_id or source_id == "src_unknown") and chunk_id and self.db:
            try:
                with self.db.session() as conn:
                    row = conn.execute(
                        """
                        SELECT d.source_id, s.url, s.title as source_title
                        FROM chunks c
                        JOIN documents d ON c.doc_id = d.doc_id
                        JOIN sources s ON d.source_id = s.source_id
                        WHERE c.chunk_id = ?
                        LIMIT 1
                        """,
                        (chunk_id,)
                    ).fetchone()
                    if row:
                        source_id = row["source_id"]
                        url = url if url != "local" else (row["url"] or "local")
                        source_title = source_title or (row["source_title"] or "")
            except Exception:
                pass

        return {
            "chunk_id": chunk_id,
            "text": text,
            "section": section,
            "page": page,
            "char_start": char_start,
            "char_end": char_end,
            "source_id": source_id,
            "url": url,
            "source_title": source_title,
            "doc_id": doc_id
        }

    def _select_chunks_for_extraction(
        self,
        state: ResearchState,
        chunks: List[Dict[str, Any]],
        limit: int = 5,
        max_per_doc: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Picks extraction chunks round-robin across compared entities, at most `max_per_doc` per document,
        so one paper cannot supply all evidence. Entities with no candidate chunk get a targeted retrieval.
        """
        entities = self._goal_subject_entities(state.goal)
        pool = list(chunks)
        seen = {c["chunk_id"] for c in pool}

        for ent in entities:
            if not any(text_mentions_entity(c.get("text", ""), ent) for c in pool):
                for rc in self.retrieve_chunks(state.session_id, f"{ent} {state.goal}", top_k=3):
                    if rc.chunk_id not in seen:
                        seen.add(rc.chunk_id)
                        pool.append(self._normalize_chunk_for_evidence(rc))

        selected: List[Dict[str, Any]] = []
        used: set = set()
        per_doc: Dict[str, int] = {}

        def take(c: Dict[str, Any], enforce_doc_cap: bool) -> bool:
            doc = c.get("doc_id") or c["chunk_id"]
            if c["chunk_id"] in used or (enforce_doc_cap and per_doc.get(doc, 0) >= max_per_doc):
                return False
            selected.append(c)
            used.add(c["chunk_id"])
            per_doc[doc] = per_doc.get(doc, 0) + 1
            return True

        # Round-robin over entities
        progress = True
        while len(selected) < limit and progress and entities:
            progress = False
            for ent in entities:
                if len(selected) >= limit:
                    break
                for c in pool:
                    if text_mentions_entity(c.get("text", ""), ent) and take(c, enforce_doc_cap=True):
                        progress = True
                        break

        # Fill remaining slots in retrieval order, still respecting the per-document cap, then relax it
        for enforce in (True, False):
            for c in pool:
                if len(selected) >= limit:
                    break
                take(c, enforce_doc_cap=enforce)

        logger.info(
            f"[{state.session_id}][EXTRACT] Selected {len(selected)} chunks from {len(per_doc)} documents "
            f"for entities {entities}."
        )
        return selected

    def run_extract_phase(
        self,
        state: ResearchState,
        chunks: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes the EXTRACT phase: extracts 3-5 atomic evidence items from chunks
        with strict quote invariance verification against source texts.
        """
        try:
            budget = self.get_budget_tracker(state.session_id)
            self.state_machine.transition(state, ResearchPhase.EXTRACT, reason="Extracting atomic evidence from chunks")

            cand_chunks = chunks or self.chunk_repo.get_by_session(state.session_id)
            if not cand_chunks:
                logger.warning(f"[{state.session_id}][EXTRACT] No chunks available for evidence extraction.")
                self.save_state(state)
                return []

            normalized_chunks = [self._normalize_chunk_for_evidence(c) for c in cand_chunks]
            normalized_chunks = self._select_chunks_for_extraction(state, normalized_chunks)

            evidence_items = self.evidence_extractor.extract_from_chunks(
                session_id=state.session_id,
                goal=state.goal,
                chunks=normalized_chunks,
                max_evidence=5,
                budget_tracker=budget
            )
            self.save_state(state)
            logger.info(f"[{state.session_id}][EXTRACT] Extracted {len(evidence_items)} verified atomic evidence items.")
            return evidence_items
        except Exception as e:
            self._handle_phase_error(state, e, "EXTRACT")
            raise

    @staticmethod
    def _normalize_citations(report: str, evidence_count: int) -> str:
        """
        Normalizes citation variants the small model produces (**E1**, (E1), E1, [E1, E2]) to [E1] form.
        Only indices within 1..evidence_count are rewritten.
        """
        if evidence_count <= 0 or not report:
            return report

        def valid(n: str) -> bool:
            return 1 <= int(n) <= evidence_count

        # [E1, E2] / [E1; E2] -> [E1][E2]
        def split_group(m):
            nums = re.findall(r"E(\d+)", m.group(0))
            return "".join(f"[E{n}]" for n in nums) if nums and all(valid(n) for n in nums) else m.group(0)
        report = re.sub(r"\[E\d+(?:\s*[,;]\s*E\d+)+\]", split_group, report)

        # [Evidence 1] / [evidence 1] -> [E1]
        report = re.sub(
            r"\[(?:Evidence|evidence)\s*(\d+)\]",
            lambda m: f"[E{m.group(1)}]" if valid(m.group(1)) else m.group(0),
            report,
        )
        # "According to Evidence 1" / "According to evidence 1" -> "According to [E1]"
        report = re.sub(
            r"\b(?:[Aa]ccording to|[Ii]n)\s+(?:Evidence|evidence)\s+(\d+)\b",
            lambda m: f"According to [E{m.group(1)}]" if valid(m.group(1)) else m.group(0),
            report,
        )
        # "Evidence 1 states" / "evidence 1 reports" -> "[E1] states"
        report = re.sub(
            r"\b(?:Evidence|evidence)\s+(\d+)\b",
            lambda m: f"[E{m.group(1)}]" if valid(m.group(1)) else m.group(0),
            report,
        )
        # **E1** / __E1__ / (E1) -> [E1]
        report = re.sub(
            r"(\*\*|__|\()E(\d+)(\*\*|__|\))",
            lambda m: f"[E{m.group(2)}]" if valid(m.group(2)) else m.group(0),
            report,
        )
        # bare E1 not already inside brackets -> [E1]
        report = re.sub(
            r"(?<![\[\w])E(\d+)(?![\]\w])",
            lambda m: f"[E{m.group(1)}]" if valid(m.group(1)) else m.group(0),
            report,
        )
        return report

    def run_basic_answer(
        self,
        state: ResearchState,
        fetched_docs: Optional[List[FetchedWebContent]] = None,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None
    ) -> str:
        """
        Synthesizes a grounded research report from verified atomic evidence and retrieved chunks.
        Enforces strict citation referencing [E1], [E2] and appends Evidence & Provenance Table.
        Transitions to PARTIAL and records missing entities if coverage is incomplete.
        """
        try:
            budget = self.get_budget_tracker(state.session_id)
            budget.assert_can_call_llm()

            core_entities = self._goal_subject_entities(state.goal)
            db_chunks = self.chunk_repo.get_by_session(state.session_id)
            cov = evaluate_coverage(core_entities, db_chunks)

            missing_entities = cov.get("missing", [])
            if missing_entities:
                state.error_message = f"Missing coverage: {', '.join(missing_entities)}"
                logger.warning(f"[{state.session_id}][ANSWER] Incomplete entity coverage: {state.error_message}")

            # Build balanced context prioritizing equal representation across core entities
            cand_chunks: List[Any] = retrieved_chunks or db_chunks
            selected_chunks: List[Any] = []

            if cand_chunks and core_entities:
                k_per_entity = max(1, (self.limits.retrieval_top_k + len(core_entities) - 1) // len(core_entities))
                used_chunk_ids = set()

                # Pass 1: Select up to k_per_entity chunks for each core entity
                for ent in core_entities:
                    ent_count = 0
                    for c in cand_chunks:
                        c_id = c.chunk_id if hasattr(c, "chunk_id") else c["chunk_id"]
                        c_text = c.text if hasattr(c, "text") else c["text"]
                        if c_id not in used_chunk_ids and check_entity_in_text(ent, c_text):
                            selected_chunks.append(c)
                            used_chunk_ids.add(c_id)
                            ent_count += 1
                            if ent_count >= k_per_entity:
                                break

                # Pass 2: Fill remaining slots up to retrieval_top_k
                for c in cand_chunks:
                    if len(selected_chunks) >= self.limits.retrieval_top_k:
                        break
                    c_id = c.chunk_id if hasattr(c, "chunk_id") else c["chunk_id"]
                    if c_id not in used_chunk_ids:
                        selected_chunks.append(c)
                        used_chunk_ids.add(c_id)
            elif cand_chunks:
                selected_chunks = list(cand_chunks[:self.limits.retrieval_top_k])

            context_snippets = []
            if selected_chunks:
                for i, c in enumerate(selected_chunks, start=1):
                    c_text = c.text if hasattr(c, "text") else c["text"]
                    c_sec = getattr(c, "section", None) or (c.get("section") if isinstance(c, dict) else "")
                    c_page = getattr(c, "page", None) or (c.get("page") if isinstance(c, dict) else "")
                    c_url = (getattr(c, "metadata", {}).get("url") if hasattr(c, "metadata") else (c.get("url") if isinstance(c, dict) else "local")) or "local"
                    sec_info = f" [Section: {c_sec}, Page: {c_page}]" if c_sec else ""
                    context_snippets.append(f"--- Evidence [{i}] ({c_url}){sec_info} ---\n{c_text}\n")
            elif fetched_docs:
                for i, doc in enumerate(fetched_docs[:5], start=1):
                    snippet = doc.text[:1200]
                    context_snippets.append(f"--- Source [{i}] ({doc.url}) ---\n{snippet}\n")

            full_context = "\n".join(context_snippets) if context_snippets else "No external documents retrieved."

            # Retrieve or eagerly extract atomic evidence items (only if not already executed in EXTRACT phase)
            evidence_items = self.evidence_repo.get_full_evidence_by_session(state.session_id)
            if not evidence_items and selected_chunks and state.phase != ResearchPhase.EXTRACT:
                normalized_selected = [self._normalize_chunk_for_evidence(c) for c in selected_chunks]
                _ = self.evidence_extractor.extract_from_chunks(
                    session_id=state.session_id,
                    goal=state.goal,
                    chunks=normalized_selected,
                    max_evidence=5,
                    budget_tracker=budget
                )
                evidence_items = self.evidence_repo.get_full_evidence_by_session(state.session_id)

            if evidence_items:
                # Plan 14.2: writer sees higher-credibility evidence first; same list backs [E#] indices below
                evidence_items = sorted(evidence_items, key=lambda ev: -(ev.get("source_score") or 0.0))
                evidence_prompt_text = CitationVerifier.format_evidence_for_prompt(evidence_items)
                prompt = (
                    f"You are a Research Analyst. Provide a clear, factual research answer to the question "
                    f"based STRICTLY on the verified atomic evidence items below.\n\n"
                    f"Question: {state.goal}\n\n"
                    f"VERIFIED EVIDENCE ITEMS:\n{evidence_prompt_text}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"Provide your response in two clear sections:\n"
                    f"1. Executive Summary: A brief factual summary addressing the question based on the evidence.\n"
                    f"2. Evidence Findings: A bulleted list where EVERY bullet begins with its citation marker [E1], [E2], etc. "
                    f"stating the key facts and numbers from that evidence item:\n"
                    f"   - [E1] (exact fact and numbers from evidence 1)\n"
                    f"   - [E2] (exact fact and numbers from evidence 2)\n\n"
                    f"CRITICAL CITATION RULES:\n"
                    f"- Every bullet in Evidence Findings MUST begin with [E1], [E2], etc.\n"
                    f"- Do not alter numbers or make ungrounded claims.\n"
                    f"- If an entity in the question is not covered in the evidence, state so in the summary.\n\n"
                    f"Synthesize the research answer now:"
                )
            else:
                prompt = (
                    f"You are a Research Analyst. Provide a clear, factual, and grounded answer to the question "
                    f"based strictly on the gathered context.\n\n"
                    f"Question: {state.goal}\n\n"
                    f"Gathered Evidence Context:\n{full_context}\n\n"
                    f"Synthesize the key findings, metrics, and comparisons directly addressing the question."
                )

            # Transition to WRITE phase
            self.state_machine.transition(state, ResearchPhase.WRITE, reason="Synthesizing report")
            res = self.llm.generate(prompt, num_predict=768)
            budget.record_llm_call(tokens=res.total_tokens, count=getattr(res, "calls_made", 1))
            raw_report = self._normalize_citations(res.content.strip(), len(evidence_items) if evidence_items else 0)

            # Append Evidence & Provenance Table and populate claims lineage if evidence items exist
            claims_json = "[]"
            supported_claims_count = 0  # Number of claims confirmed SUPPORTED by NLI (verified: True)
            cited_claims_count = 0      # Number of claims with valid [E#] citation and numeric consistency
            numeric_mismatch_claims_count = 0
            uncited_numeric_claims_count = 0
            contradicted_claims_count = 0
            off_topic_claims_count = 0
            nli_llm_evaluated_count = 0
            goal_core_entities = self._goal_subject_entities(state.goal)

            if evidence_items:
                augmented_answer, citation_stats = CitationVerifier.verify_and_append_appendix(raw_report, evidence_items)
                answer = augmented_answer
                valid_indices = citation_stats.get("valid_indices", [])
                claims_json = json.dumps(valid_indices)

                # Claim Lineage: Extract substantive sentences from the report body
                # Each claim in the claims table MUST originate from a sentence in the report!
                clean_body = raw_report.split("### Evidence & Provenance Table")[0].strip()
                # Fold trailing citation lines like "- Source: [E1]" or "Source: [E1]" into preceding statement
                clean_body = re.sub(
                    r"\n\s*[-*]?\s*(?:Source|Ref|Reference)s?:\s*(\[E\d+(?:[,\s]+E\d+)*\])",
                    r" \1",
                    clean_body,
                    flags=re.IGNORECASE
                )
                report_sentences = [
                    s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", clean_body)
                    if s.strip() and len(s.strip()) >= 10
                ]

                for sent in report_sentences:
                    # Ignore markdown headers, footers, tables, or formatting lines
                    trimmed = sent.strip()
                    if trimmed.startswith("#") or trimmed.startswith("---") or trimmed.startswith("|"):
                        continue
                    lower_sent = trimmed.lower()
                    clean_lower = re.sub(r"^[-*\d.\s]+", "", lower_sent)
                    if (
                        clean_lower.startswith(("source:", "sources:", "url:", "urls:", "ref:", "reference:", "references:", "quote:", "quotes:"))
                        or lower_sent.startswith(("http://", "https://"))
                        or lower_sent.startswith(("**citation", "**source", "**reference"))
                    ):
                        continue

                    cited_tokens = re.findall(r"\[E(\d+)\]", sent)
                    valid_cites_in_sent = [int(t) for t in cited_tokens if 1 <= int(t) <= len(evidence_items)]
                    substantive_numbers = extract_substantive_numbers(sent)

                    if valid_cites_in_sent:
                        cited_evs = [evidence_items[c_idx - 1] for c_idx in valid_cites_in_sent]
                        combined_quotes = " ".join(
                            f"{ev.get('exact_quote', '')} {ev.get('raw_quote', '')}" for ev in cited_evs
                        )

                        # Plan 24: Numeric Verification for Cited Sentences
                        # Check that every substantive number in the sentence appears in at least one cited quote
                        mismatched_numbers = []
                        for num in substantive_numbers:
                            num_pattern = rf"(?<![a-zA-Z0-9_.])(?<!\d){re.escape(num)}(?!\d)(?![a-zA-Z0-9_.])"
                            if not re.search(num_pattern, combined_quotes) and num not in combined_quotes:
                                mismatched_numbers.append(num)

                        claim_id = f"clm_{uuid.uuid4().hex[:8]}"
                        if mismatched_numbers:
                            # Numeric Mismatch detected in cited sentence!
                            self.claim_repo.add(
                                claim_id=claim_id,
                                session_id=state.session_id,
                                text=sent,
                                evidence_ids=[ev.get("evidence_id") for ev in cited_evs],
                                verification={
                                    "verified": False,
                                    "numeric_match": False,
                                    "citation_indices": valid_cites_in_sent,
                                    "citation_index": valid_cites_in_sent[0],
                                    "mismatched_numbers": mismatched_numbers,
                                    "reason": f"Numbers {mismatched_numbers} not found in cited evidence quotes"
                                },
                                status="NUMERIC_MISMATCH"
                            )
                            numeric_mismatch_claims_count += 1
                        else:
                            # Off-topic Claim Guardrail (run BEFORE NLI to preserve LLM budget):
                            # Stricter rule:
                            # 1. Direct mention in claim -> ON_TOPIC
                            # 2. Claim mentions other models without goal models -> OFF_TOPIC (even if evidence belongs to goal!)
                            # 3. Only pronoun claims use evidence subject.
                            is_on_topic = self._is_claim_on_topic(sent, cited_evs, goal_core_entities)

                            if not is_on_topic:
                                claim_status = "OFF_TOPIC"
                                is_verified = False
                                off_topic_claims_count += 1
                                cited_claims_count += 1
                                logger.info(
                                    f"[{state.session_id}][WRITE] Claim marked OFF_TOPIC (not on goal entities {goal_core_entities}): '{sent[:80]}...'"
                                )
                                self.claim_repo.add(
                                    claim_id=claim_id,
                                    session_id=state.session_id,
                                    text=sent,
                                    evidence_ids=[ev.get("evidence_id") for ev in cited_evs],
                                    verification={
                                        "verified": False,
                                        "is_on_topic": False,
                                        "entailment": NLILabel.NOT_SUPPORTED.value,
                                        "entailment_confidence": 1.0,
                                        "entailment_reason": "Claim does not address any goal core entity; NLI evaluation skipped to preserve budget",
                                        "verifier_type": "guardrail",
                                        "nli_prompt_version": NLI_PROMPT_VERSION,
                                        "numeric_match": True,
                                        "citation_indices": valid_cites_in_sent,
                                        "citation_index": valid_cites_in_sent[0],
                                        "quote": cited_evs[0].get("exact_quote"),
                                        "source_id": cited_evs[0].get("source_id")
                                    },
                                    status="OFF_TOPIC"
                                )
                            else:
                                # Cited, numerically consistent, and on-topic -> Plan 23 NLI Entailment Verification
                                quotes_for_nli = [
                                    ev.get("exact_quote") or ev.get("raw_quote") or "" for ev in cited_evs
                                    if (ev.get("exact_quote") or ev.get("raw_quote"))
                                ]
                                nli_res = self.claim_verification_pipeline.verify_claim_against_quotes(
                                    sent, quotes_for_nli, budget_tracker=budget, known_entities=goal_core_entities
                                )
                                if nli_res.verifier_type == "llm":
                                    nli_llm_evaluated_count += 1

                                claim_status = "CITED"
                                is_verified = (nli_res.label == NLILabel.SUPPORTED)
                                if nli_res.label == NLILabel.SUPPORTED:
                                    supported_claims_count += 1
                                if nli_res.label == NLILabel.CONTRADICTED:
                                    contradicted_claims_count += 1

                                cited_claims_count += 1

                                self.claim_repo.add(
                                    claim_id=claim_id,
                                    session_id=state.session_id,
                                    text=sent,
                                    evidence_ids=[ev.get("evidence_id") for ev in cited_evs],
                                    verification={
                                        "verified": is_verified,
                                        "is_on_topic": True,
                                        "entailment": nli_res.label.value,
                                        "entailment_confidence": nli_res.confidence,
                                        "entailment_reason": nli_res.reason,
                                        "verifier_type": nli_res.verifier_type,
                                        "nli_prompt_version": NLI_PROMPT_VERSION,
                                        "numeric_match": True,
                                        "citation_indices": valid_cites_in_sent,
                                        "citation_index": valid_cites_in_sent[0],
                                        "quote": cited_evs[0].get("exact_quote"),
                                        "source_id": cited_evs[0].get("source_id")
                                    },
                                    status=claim_status
                                )
                    elif substantive_numbers:
                        # Sentence contains substantive metrics/numbers but has NO citation!
                        claim_id = f"clm_{uuid.uuid4().hex[:8]}"
                        self.claim_repo.add(
                            claim_id=claim_id,
                            session_id=state.session_id,
                            text=sent,  # The actual uncited sentence from the report!
                            evidence_ids=[],
                            verification={
                                "verified": False,
                                "numeric_match": False,
                                "unsupported_numbers": substantive_numbers,
                                "reason": "Uncited substantive numeric claim"
                            },
                            status="UNSUPPORTED"
                        )
                        uncited_numeric_claims_count += 1
            else:
                answer = raw_report

            # Evidence-based entity coverage check:
            # Each core entity in research goal must have at least 1 verified atomic evidence item!
            goal_core_entities = self._goal_subject_entities(state.goal)

            ev_coverage = evaluate_evidence_coverage(goal_core_entities, evidence_items)
            missing_evidence_entities = ev_coverage["missing"] if len(goal_core_entities) > 1 else []

            # Transition state to DONE (or PARTIAL if missing coverage, 0 evidence, uncited/mismatched numbers, or insufficient info)
            insufficient_phrases = [
                "no information available",
                "insufficient information",
                "not enough information",
                "cannot be answered",
                "unable to find information",
                "no evidence found",
                "there is no information"
            ]
            answer_lower = answer.lower()
            is_insufficient = any(phrase in answer_lower for phrase in insufficient_phrases)

            total_unsupported_numeric = numeric_mismatch_claims_count + uncited_numeric_claims_count
            has_valid_evidence = (
                bool(context_snippets or evidence_items)
                and bool(state.source_ids)
                and "no external documents retrieved" not in full_context.lower()
                and not is_insufficient
                and not bool(missing_entities)
                and not bool(missing_evidence_entities)
                and len(evidence_items) > 0
                and supported_claims_count > 0
                and total_unsupported_numeric == 0
                and contradicted_claims_count == 0
            )

            # Record transparent reasons if transitioning to PARTIAL
            if not has_valid_evidence:
                partial_reasons = []
                if len(evidence_items) == 0:
                    partial_reasons.append("No verified atomic evidence extracted")
                if missing_evidence_entities:
                    partial_reasons.append(f"Missing evidence for: {', '.join(missing_evidence_entities)}")
                if cited_claims_count == 0 and numeric_mismatch_claims_count == 0 and evidence_items:
                    partial_reasons.append("Report contains no valid citations")
                elif off_topic_claims_count > 0 and supported_claims_count == 0:
                    partial_reasons.append("Report claims do not address the research goal entities (off-topic)")
                elif cited_claims_count > 0 and supported_claims_count == 0 and numeric_mismatch_claims_count == 0:
                    if nli_llm_evaluated_count == 0:
                        partial_reasons.append("NLI unavailable – claims not verified")
                    else:
                        partial_reasons.append("Report contains cited claims but none were confirmed as fully supported by evidence")
                if numeric_mismatch_claims_count > 0:
                    partial_reasons.append("Report contains numeric claims mismatched with cited evidence")
                if uncited_numeric_claims_count > 0:
                    partial_reasons.append("Report contains unsupported numeric claims without citations")
                if contradicted_claims_count > 0:
                    partial_reasons.append("Report contains claims contradicted by cited evidence")
                if is_insufficient:
                    partial_reasons.append("Insufficient information in gathered context")

                if partial_reasons:
                    reason_msg = "; ".join(partial_reasons)
                    if not state.error_message:
                        state.error_message = reason_msg
                    elif reason_msg not in state.error_message:
                        state.error_message = f"{state.error_message}; {reason_msg}"

            target_phase = ResearchPhase.DONE if has_valid_evidence else ResearchPhase.PARTIAL
            self.state_machine.transition(state, target_phase, reason="Report synthesis completed")
            self.save_state(state)

            # Save report
            report_id = f"rep_{uuid.uuid4().hex[:8]}"
            self.report_repo.create(
                report_id=report_id,
                session_id=state.session_id,
                title=f"Research: {state.goal[:60]}",
                content_markdown=answer,
                claims_json=claims_json,
                status=state.status.value
            )

            # Save final trajectory into 'raw' partition with verified=False (resolves P0 fake confidence)
            self.trajectory_repo.add(
                trajectory_id=f"traj_{uuid.uuid4().hex[:12]}",
                session_id=state.session_id,
                task_type="answer_synthesis",
                model_source=self._get_model_source(),
                payload={"goal": state.goal, "answer": answer, "sources_count": len(state.source_ids)},
                verified=False,
                quality_score=0.0,
                partition="raw"
            )

            logger.info(f"[{state.session_id}][ANSWER] Final report saved (report_id={report_id}, status={state.status.value}).")
            return answer
        except Exception as e:
            self._handle_phase_error(state, e, "WRITE")
            raise

    def verify_session_claims(self, session_id: str, budget_tracker: Optional[ExecutionBudgetTracker] = None) -> List[Dict[str, Any]]:
        """
        Week 4 Plan 23: Entailment Verification across all claims in a session.
        Takes claims currently in CITED status, checks semantic entailment against cited evidence quotes,
        and updates claim records with verified status, entailment label, confidence, and reason.
        Never overwrites NUMERIC_MISMATCH or UNSUPPORTED claims. Status is kept per numeric rules.
        """
        claims = self.claim_repo.get_by_session(session_id)
        evidence_map = {e["evidence_id"]: e for e in self.evidence_repo.get_full_evidence_by_session(session_id)}
        updated_claims = []

        sess = self.session_repo.get(session_id)
        goal = sess.get("goal") if sess else ""
        known_entities = self._goal_subject_entities(goal) if goal else None

        for claim in claims:
            # DO NOT touch NUMERIC_MISMATCH, UNSUPPORTED, or OFF_TOPIC claims! (Review item 3)
            if claim.get("status") in ("NUMERIC_MISMATCH", "UNSUPPORTED", "OFF_TOPIC"):
                continue

            ev_ids = claim.get("evidence_ids") or []
            if not ev_ids:
                continue

            quotes = []
            for eid in ev_ids:
                ev = evidence_map.get(eid)
                if ev:
                    quote = ev.get("exact_quote") or ev.get("raw_quote") or ""
                    if quote.strip():
                        quotes.append(quote.strip())

            if not quotes:
                continue

            current_verif = claim.get("verification") or {}
            numeric_match = current_verif.get("numeric_match", True)
            sent = claim.get("text", "")

            # Off-topic check (checked BEFORE NLI to preserve LLM budget and handles pronouns via cited evidence subjects)
            cited_ev_items = [evidence_map[eid] for eid in ev_ids if eid in evidence_map]
            is_on_topic = self._is_claim_on_topic(sent, cited_ev_items, known_entities)

            if not is_on_topic:
                current_verif.update({
                    "is_on_topic": False,
                    "verified": False,
                    "entailment": NLILabel.NOT_SUPPORTED.value,
                    "entailment_confidence": 1.0,
                    "entailment_reason": "Claim does not address any goal core entity; NLI evaluation skipped to preserve budget",
                    "verifier_type": "guardrail",
                    "nli_prompt_version": NLI_PROMPT_VERSION
                })
                self.claim_repo.update_verification(
                    claim_id=claim["claim_id"],
                    verification=current_verif,
                    status="OFF_TOPIC"
                )
                updated = self.claim_repo.get_by_id(claim["claim_id"])
                if updated:
                    updated_claims.append(updated)
                continue

            # On-topic -> run NLI verification
            res = self.claim_verification_pipeline.verify_claim_against_quotes(
                claim["text"], quotes, budget_tracker=budget_tracker, known_entities=known_entities
            )
            is_verified = (res.label == NLILabel.SUPPORTED and numeric_match)

            current_verif.update({
                "entailment": res.label.value,
                "entailment_confidence": res.confidence,
                "entailment_reason": res.reason,
                "verifier_type": res.verifier_type,
                "nli_prompt_version": NLI_PROMPT_VERSION,
                "is_on_topic": True,
                "verified": is_verified
            })

            # Keep status according to numeric rules (CITED)
            self.claim_repo.update_verification(
                claim_id=claim["claim_id"],
                verification=current_verif,
                status=claim.get("status", "CITED")
            )
            updated = self.claim_repo.get_by_id(claim["claim_id"])
            if updated:
                updated_claims.append(updated)

        return updated_claims

    def run_week1(self, session_or_goal: Union[ResearchState, str]) -> Dict[str, Any]:
        """
        End-to-End Orchestrator with real entity coverage check and iterative research loop:
        Question -> PLAN -> (SEARCH -> FETCH -> CLEAN -> RETRIEVE -> EVALUATE)* -> WRITE -> DONE / PARTIAL
        """
        if isinstance(session_or_goal, str):
            if session_or_goal.startswith("sess_") and self.session_repo.get(session_or_goal):
                state = self.load_state(session_or_goal)
            else:
                state = self.create_session(goal=session_or_goal)
        else:
            state = session_or_goal

        budget = self.get_budget_tracker(state.session_id)
        logger.info(f"[{state.session_id}][E2E] Starting end-to-end research for: '{state.goal}'")

        try:
            # 1. Plan
            self.run_plan_phase(state)

            docs: List[FetchedWebContent] = []
            top_chunks: List[RetrievedChunk] = []

            # Research Loop (enforcing max_research_steps and evaluate_next_step)
            while True:
                # Search: use open questions as targeted custom queries if available
                search_queries = None
                if state.open_questions:
                    search_queries = [oq.text for oq in state.open_questions]
                    logger.info(f"[{state.session_id}][SEARCH] Targeted search queries from open questions: {search_queries}")

                urls = self.run_search_phase(state, custom_queries=search_queries)

                # Fetch
                new_docs = self.run_fetch_phase(state, urls=urls)
                docs.extend(new_docs)

                # Clean & Chunk
                self.run_clean_phase(state)

                # Hybrid Retrieval
                top_chunks = self.run_retrieve_phase(state, query=state.goal)

                # Evaluate: Transition to EVALUATE
                self.state_machine.transition(state, ResearchPhase.EVALUATE, reason="Evaluating research iteration")

                # Core Entity Coverage Check (compared subjects only; datasets/benchmarks are context)
                core_entities = self._goal_subject_entities(state.goal)
                db_chunks = self.chunk_repo.get_by_session(state.session_id)
                cov = evaluate_coverage(core_entities, db_chunks)
                logger.info(f"[{state.session_id}][EVALUATE] Entity coverage: {cov}")

                # Update state.open_questions:
                # 1. Filter out open questions for entities that are now covered
                updated_open_questions = [
                    oq for oq in state.open_questions
                    if oq.derived_from_gap in cov["missing"]
                ]
                # 2. Add open questions for missing entities not yet tracked
                tracked_gaps = {oq.derived_from_gap for oq in updated_open_questions}
                for missing_ent in cov["missing"]:
                    if missing_ent not in tracked_gaps:
                        context_keywords = [
                            w for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", state.goal.lower())
                            if w not in [e.lower() for e in core_entities if e != missing_ent]
                            and w != missing_ent.lower()
                            and w not in {"compare", "comparison", "vs", "versus", "between", "and", "the", "for", "with", "from"}
                        ]
                        ctx_str = " ".join(context_keywords[:2])
                        q_text = f"{missing_ent} {ctx_str} benchmark".strip() if ctx_str else f"{missing_ent} benchmark"
                        updated_open_questions.append(
                            OpenQuestion(
                                question_id=f"oq_{uuid.uuid4().hex[:8]}",
                                text=q_text,
                                priority=1,
                                derived_from_gap=missing_ent
                            )
                        )
                state.open_questions = updated_open_questions

                next_phase = self.state_machine.evaluate_next_step(state)  # Increments state.step!
                budget.step_count = state.step  # Synchronize budget tracker counter!
                self.save_state(state)

                logger.info(f"[{state.session_id}][EVALUATE] Step {state.step}/{self.limits.max_research_steps}, next: {next_phase.value}")

                if next_phase == ResearchPhase.SEARCH and state.open_questions and budget.can_search():
                    continue
                else:
                    break

            # 4. Extract atomic evidence & Answer
            if top_chunks:
                try:
                    self.run_extract_phase(state, chunks=top_chunks)
                except Exception as extract_err:
                    logger.warning(f"[{state.session_id}][E2E] Extract phase encountered issue ({extract_err}), continuing to WRITE.")

            answer = self.run_basic_answer(state, fetched_docs=docs, retrieved_chunks=top_chunks)

            return {
                "session_id": state.session_id,
                "goal": state.goal,
                "phase": state.phase.value,
                "status": state.status.value,
                "step": state.step,
                "plan": state.plan.model_dump() if state.plan else None,
                "visited_queries": state.visited_queries,
                "sources_count": len(state.source_ids),
                "documents_fetched": len(docs),
                "chunks_count": self.chunk_repo.count_by_session(state.session_id),
                "evidence_count": len(self.evidence_repo.get_by_session(state.session_id)),
                "answer": answer,
                "error_message": state.error_message,
                "budget": budget.summary()
            }
        except Exception as e:
            self._handle_phase_error(state, e, "RUN_E2E")
            raise
