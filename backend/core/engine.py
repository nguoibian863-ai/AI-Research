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
    GENERIC_RESEARCH_TERMS
)

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
        hybrid_retriever: Optional[HybridRetriever] = None
    ):
        self.llm = llm
        self.db = db
        self.limits = limits or ResearchLimits()
        self.state_machine = StateMachine(limits=self.limits)
        self.search_tool = search_tool or WebSearchTool(max_results=self.limits.max_search_results)
        self.fetch_tool = fetch_tool or WebFetchTool()
        self.pdf_tool = pdf_tool or PDFFetchTool()
        self.chunker = chunker or SectionAwareChunker(max_chunk_tokens=self.limits.max_chunk_tokens)
        self.embedding_backend = embedding_backend or LocalHashEmbeddingBackend()
        
        # Per-session hybrid retrievers (BM25 + FAISS isolation)

        self._session_retrievers: Dict[str, HybridRetriever] = {}
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

        # Session Budget Trackers (cached in-memory, backed by SQLite)
        self._budget_trackers: Dict[str, ExecutionBudgetTracker] = {}

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

    def retrieve_chunks(self, session_id: str, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Read-only chunk retrieval without mutating session state or phase."""
        chunks = self.chunk_repo.get_by_session(session_id)
        if not chunks:
            return []
        retriever = self.get_session_retriever(session_id)
        retriever.index_chunks(chunks)
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
                f"hardware specifications, datasets, and baseline comparisons.\n\n"
                f"Example Benchmark Plan:\n"
                f"Goal: Compare RocksDB vs LevelDB write amplification and throughput on NVMe SSDs\n"
                f"Tasks:\n"
                f"1. Investigate RocksDB LSM-tree compaction strategies, write stall mitigation, and random write throughput.\n"
                f"2. Investigate LevelDB write amplification factor, memory table architecture, and sequential read/write benchmarks.\n"
                f"3. Compare CPU overhead, space amplification, and P99 latency on high-iops NVMe storage.\n\n"
                f"Question: {state.goal}"
            )
            res = self.llm.structured_generate(prompt, schema=ResearchPlanSchema)
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
                model_source=getattr(self.llm, "model", "mock"),
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
            if not queries_to_run:
                budget.assert_can_call_llm()
                prompt = (
                    f"Based on the research goal and plan, generate 2-4 concise, highly-targeted keyword search queries "
                    f"(3 to 8 words per query). Do NOT use conversational question sentences.\n\n"
                    f"Good Examples:\n"
                    f"- 'RocksDB LevelDB write amplification benchmark NVMe'\n"
                    f"- 'RocksDB compaction throughput latency SSD'\n"
                    f"- 'LevelDB LSM tree space amplification evaluation'\n\n"
                    f"Goal: {state.goal}\n"
                    f"Plan: {state.plan.model_dump_json() if state.plan else ''}\n"
                    f"Previously visited: {state.visited_queries}"
                )
                res = self.llm.structured_generate(prompt, schema=GeneratedQueriesSchema)
                budget.record_llm_call(tokens=res.total_tokens, count=getattr(res, "calls_made", 1))
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

                        if url not in existing_urls and url not in found_urls:
                            found_urls.append(url)
                            existing_urls.add(url)
                            source_id = f"src_{uuid.uuid4().hex[:8]}"
                            state.source_ids.append(source_id)
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
                            canonical_key=url.lower().rstrip("/"),
                            authors=authors_list,
                            published_at=pub_date,
                            source_type="pdf" if is_pdf_url else "web"
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

            # 2. Entity-level queries for balanced coverage across distinct subjects
            core_entities = extract_core_entities(state.goal)
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

    def run_basic_answer(
        self,
        state: ResearchState,
        fetched_docs: Optional[List[FetchedWebContent]] = None,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None
    ) -> str:
        """
        Synthesizes a grounded initial answer from retrieved chunks or fetched sources.
        Guarantees balanced context representation across core entities.
        Transitions to PARTIAL and records missing entities if coverage is incomplete.
        """
        try:
            budget = self.get_budget_tracker(state.session_id)
            budget.assert_can_call_llm()

            core_entities = extract_core_entities(state.goal)
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
            budget.record_llm_call(tokens=res.total_tokens, count=getattr(res, "calls_made", 1))
            answer = res.content.strip()

            # Transition state to DONE (or PARTIAL if missing coverage or insufficient info)
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

            has_valid_evidence = (
                bool(context_snippets)
                and bool(state.source_ids)
                and "no external documents retrieved" not in full_context.lower()
                and not is_insufficient
                and not bool(missing_entities)
            )
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
                status=state.status.value
            )

            # Save final trajectory into 'raw' partition with verified=False (resolves P0 fake confidence)
            self.trajectory_repo.add(
                trajectory_id=f"traj_{uuid.uuid4().hex[:12]}",
                session_id=state.session_id,
                task_type="answer_synthesis",
                model_source=getattr(self.llm, "model", "mock"),
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

                # Core Entity Coverage Check
                core_entities = extract_core_entities(state.goal)
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

            # 4. Answer
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
                "answer": answer,
                "budget": budget.summary()
            }
        except Exception as e:
            self._handle_phase_error(state, e, "RUN_E2E")
            raise
