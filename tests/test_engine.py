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


def test_engine_routes_pdf_and_stores_metadata(tmp_path):
    """
    P0 Test: PDF routing and author/date metadata storage.
    Verifies that URLs ending in .pdf route to PDFFetchTool, and authors/dates
    are recorded in the SQLite sources table.
    """
    import json
    import pymupdf
    from backend.tools.pdf_fetch import PDFFetchTool

    pdf_file = tmp_path / "paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "1. Abstract\nAutonomous 3D detection paper by Dr. LiDAR.")
    doc.set_metadata({"title": "3D Detection Paper", "author": "Dr. LiDAR", "creationDate": "2026-03-01"})
    doc.save(str(pdf_file))
    doc.close()

    db_path = tmp_path / "pdf_test.db"
    db = DatabaseManager(db_path=db_path)
    engine = ResearchEngine(
        llm=MockLLMBackend(),
        db=db,
        pdf_tool=PDFFetchTool(cache_dir=tmp_path / "cache")
    )

    state = engine.create_session("Test PDF routing")
    state.phase = ResearchPhase.SEARCH
    engine.save_state(state)

    # Fetch local PDF path which ends in .pdf
    docs = engine.run_fetch_phase(state, urls=[str(pdf_file)])
    assert len(docs) == 1
    assert docs[0].title == "3D Detection Paper"

    # Verify source was recorded with author and type "pdf"
    sources = engine.source_repo.get_by_session(state.session_id)
    assert len(sources) == 1
    src = sources[0]
    assert src["source_type"] == "pdf"
    authors = json.loads(src["authors_json"])
    assert "Dr. LiDAR" in authors


def test_engine_enforces_max_total_chunks(tmp_path):
    """
    P1 Test: Hard Limit enforcement of max_total_chunks.
    Verifies that chunks in SQLite never exceed limits.max_total_chunks.
    """
    db = DatabaseManager(db_path=tmp_path / "chunks_limit.db")
    engine = ResearchEngine(
        llm=MockLLMBackend(),
        db=db,
        limits=ResearchLimits(max_total_chunks=3, max_chunk_tokens=20)
    )

    state = engine.create_session("Test chunk limit")
    sid = state.session_id

    # Add source and doc with many paragraphs
    src_id = "src_lim_1"
    doc_id = "doc_lim_1"
    engine.source_repo.add(source_id=src_id, session_id=sid, url="https://example.com/long", title="Long", domain="example.com")
    engine.doc_repo.add(
        doc_id=doc_id, source_id=src_id, file_path=None, content_hash="h1",
        raw_text="\n\n".join([f"Paragraph {i} has detailed content to create multiple chunks." for i in range(10)])
    )

    state.phase = ResearchPhase.FETCH
    engine.save_state(state)

    total_chunks = engine.run_clean_phase(state)
    assert total_chunks == 3  # Capped at max_total_chunks

    db_chunks = engine.chunk_repo.get_by_session(sid)
    assert len(db_chunks) == 3


def test_engine_search_skips_blocked_domains(tmp_path):
    """
    P1 Test: Search domain blocklist with substring-safe matching.
    Verifies that social media / spam domains (x.com, m.facebook.com) are filtered,
    while valid domains containing blocked substrings (e.g. dropbox.com containing 'x.com')
    are NOT incorrectly blocked.
    """
    from backend.tools.web_search import SearchResultItem

    class BlocklistTestSearchTool:
        def search(self, query: str, max_results: int = 5):
            return [
                SearchResultItem(title="Facebook Spam", url="https://m.facebook.com/group/ai", snippet="spam on facebook"),
                SearchResultItem(title="Real Tech Blog", url="https://towardsdatascience.com/3d-lidar", snippet="real tech blog benchmark"),
                SearchResultItem(title="Twitter post", url="https://x.com/user/status/123", snippet="tweet on x.com"),
                SearchResultItem(title="Dropbox Tech Paper", url="https://dropbox.com/s/benchmark.pdf", snippet="whitepaper on benchmark evaluation")
            ]

    db = DatabaseManager(db_path=tmp_path / "blocklist.db")
    engine = ResearchEngine(
        llm=MockLLMBackend(),
        db=db,
        search_tool=BlocklistTestSearchTool()
    )

    state = engine.create_session("Evaluation benchmark test")
    state.phase = ResearchPhase.PLAN
    engine.save_state(state)

    found_urls = engine.run_search_phase(state, custom_queries=["benchmark evaluation"])
    # towardsdatascience.com and dropbox.com must be retained; x.com and m.facebook.com must be dropped
    assert len(found_urls) == 2
    assert any("towardsdatascience.com" in u for u in found_urls)
    assert any("dropbox.com" in u for u in found_urls)
    assert not any(u.split("/")[2] in {"x.com", "m.facebook.com"} for u in found_urls)

    sources = engine.source_repo.get_by_session(state.session_id)
    domains = [s["domain"] for s in sources]
    assert "towardsdatascience.com" in domains
    assert "dropbox.com" in domains
    assert "m.facebook.com" not in domains
    assert "x.com" not in domains


def test_engine_search_relevance_whole_words_and_core_entities(tmp_path):
    """
    Issue #1 Regression Test (Neutral domain: RocksDB vs LevelDB storage):
    Search relevance filtering must:
    1. Match discrete whole words (e.g., 'map' must NOT match 'sitemap', 'nds' must NOT match 'friends').
    2. Focus on core domain entities of goal/query, ignoring generic research terms ('benchmark', 'accuracy', 'write').
    """
    from backend.tools.web_search import SearchResultItem

    class RelevanceSearchTool:
        def search(self, query: str, max_results: int = 5):
            return [
                SearchResultItem(
                    title="Store Navigation",
                    url="https://store.example.com/sitemap.xml",
                    snippet="Download the sitemap for our online retail store."
                ),
                SearchResultItem(
                    title="Social Status",
                    url="https://quotes.example.com/status",
                    snippet="Best friends forever - WhatsApp status quotes."
                ),
                SearchResultItem(
                    title="Storage Hardware Review",
                    url="https://hardware.example.com/review",
                    snippet="Top 10 benchmark storage devices 2026 performance review."
                ),
                SearchResultItem(
                    title="RocksDB Storage Engine Architecture",
                    url="https://rocksdb.org/docs/benchmark",
                    snippet="Comprehensive RocksDB throughput and write amplification benchmark on NVMe SSD."
                )
            ]

    db = DatabaseManager(db_path=tmp_path / "relevance.db")
    engine = ResearchEngine(
        llm=MockLLMBackend(),
        db=db,
        search_tool=RelevanceSearchTool()
    )

    state = engine.create_session("Compare RocksDB and LevelDB write amplification on NVMe")
    state.phase = ResearchPhase.PLAN
    engine.save_state(state)

    found_urls = engine.run_search_phase(state, custom_queries=["rocksdb nvme write amplification"])

    # Only rocksdb.org doc must be retained!
    # sitemap, friends, and generic hardware review must all be discarded!
    assert len(found_urls) == 1
    assert "rocksdb.org" in found_urls[0]


def test_entity_coverage_loop_and_partial_reporting(tmp_path):
    """
    P0 Test: Verifies that comparison questions (A vs B) trigger targeted queries
    in EVALUATE when entity B is missing, and if B remains uncovered at max_steps,
    the session cleanly transitions to PARTIAL with 'Missing coverage: ...'.
    """
    from backend.tools.web_search import SearchResultItem
    from backend.tools.web_fetch import FetchedWebContent
    from backend.retrieval.embeddings import LocalHashEmbeddingBackend

    searched_queries = []

    class MockSearch:
        def search(self, query: str, max_results: int = 5):
            searched_queries.append(query)
            if "redis" in query.lower():
                return [
                    SearchResultItem(
                        title="Redis In-Memory Data Store",
                        url="https://redis.io/docs/benchmark",
                        snippet="Redis latency and throughput benchmarks in RAM.",
                        query=query,
                        rank=1
                    )
                ]
            # No results for memcached
            return []

    class MockFetch:
        def fetch(self, url: str):
            return FetchedWebContent(
                url=url,
                title="Redis Benchmarks",
                text="Redis achieves 100k ops/sec throughput with sub-millisecond latency.",
                content_hash=f"hash_{hash(url)}"
            )

    canned = {
        "ResearchPlanSchema": {
            "goal": "Compare Redis and Memcached throughput",
            "tasks": [
                {"task_id": "t1", "description": "Benchmark Redis", "expected_evidence": "ops/sec"},
                {"task_id": "t2", "description": "Benchmark Memcached", "expected_evidence": "ops/sec"}
            ],
            "key_hypotheses": ["Redis has higher throughput"]
        },
        "GeneratedQueriesSchema": {
            "queries": [
                {"query": "Redis throughput benchmark", "query_type": "evidence", "rationale": "Redis data"}
            ]
        }
    }

    db = DatabaseManager(db_path=tmp_path / "coverage_test.db")
    engine = ResearchEngine(
        llm=MockLLMBackend(canned_responses=canned),
        db=db,
        limits=ResearchLimits(max_research_steps=2),
        search_tool=MockSearch(),
        fetch_tool=MockFetch(),
        embedding_backend=LocalHashEmbeddingBackend(dimension=64)
    )

    state = engine.create_session("Compare Redis and Memcached throughput")
    result = engine.run_week1(state)

    # 1. State must end in PARTIAL because Memcached was never found
    assert result["phase"] == "PARTIAL"
    assert result["status"] == "PARTIAL"
    assert "Missing coverage: memcached" in (state.error_message or "")

    # 2. Targeted search query for the missing entity 'memcached' was executed in step 2
    assert any("memcached" in q.lower() for q in searched_queries)


def test_entity_balanced_context_generation(tmp_path):
    """
    P0 Test: Verifies entity-balanced context generation in run_basic_answer:
    Even if Entity A has 10 chunks and Entity B has only 1 chunk, the context
    sent to the LLM synthesizer is guaranteed to include Entity B.
    """
    db = DatabaseManager(db_path=tmp_path / "balanced_test.db")
    mock_llm = MockLLMBackend()
    engine = ResearchEngine(
        llm=mock_llm,
        db=db,
        limits=ResearchLimits(retrieval_top_k=4)
    )

    state = engine.create_session("Compare SystemAlpha and SystemBeta performance")
    engine.source_repo.add(
        source_id="src_1",
        session_id=state.session_id,
        url="https://alpha.example.com",
        title="SystemAlpha Docs",
        domain="alpha.example.com",
        canonical_key="alpha"
    )
    state.source_ids.append("src_1")

    engine.doc_repo.add(
        doc_id="doc_alpha",
        source_id="src_1",
        file_path=None,
        content_hash="hash_alpha",
        raw_text="SystemAlpha full raw text"
    )

    # Add 10 chunks for SystemAlpha
    alpha_chunks = [
        {
            "chunk_id": f"chunk_alpha_{i}",
            "doc_id": "doc_alpha",
            "text": f"SystemAlpha metric {i}: throughput is {1000 + i} req/sec with low latency.",
            "page": None,
            "section": "Metrics",
            "char_start": 0,
            "char_end": 50,
            "token_count": 10
        }
        for i in range(10)
    ]
    engine.chunk_repo.add_batch(alpha_chunks)

    # Add 1 chunk for SystemBeta
    beta_chunks = [
        {
            "chunk_id": "chunk_beta_1",
            "doc_id": "doc_alpha",
            "text": "SystemBeta metric: throughput is 950 req/sec with high stability.",
            "page": None,
            "section": "Metrics",
            "char_start": 0,
            "char_end": 50,
            "token_count": 10
        }
    ]
    engine.chunk_repo.add_batch(beta_chunks)

    state.phase = ResearchPhase.EVALUATE
    engine.save_state(state)
    _ = engine.run_basic_answer(state)

    assert len(mock_llm.call_history) == 1
    prompt_sent = mock_llm.call_history[0]["prompt"]
    assert "SystemAlpha metric" in prompt_sent
    assert "SystemBeta metric" in prompt_sent



