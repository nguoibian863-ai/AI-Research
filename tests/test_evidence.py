import pytest
from pathlib import Path
from backend.evidence.extractor import find_quote_in_text, EvidenceExtractor
from backend.evidence.verifier import CitationVerifier
from backend.db.database import DatabaseManager
from backend.llm.mock import MockLLMBackend
from backend.core.state import ResearchPhase


def test_find_quote_in_text_variations():
    text = (
        "In our comprehensive benchmarks, YOLOv8n achieves a 37.3 mAP on COCO val2017\n"
        "with an inference latency of 0.99 ms on NVIDIA A100 TensorRT."
    )

    # 1. Exact match
    res1 = find_quote_in_text("37.3 mAP on COCO val2017", text)
    assert res1 is not None
    start, end, matched = res1
    assert matched == "37.3 mAP on COCO val2017"
    assert text[start:end] == matched

    # 2. Case-insensitive match
    res2 = find_quote_in_text("yolov8n achieves a 37.3 map", text)
    assert res2 is not None
    start, end, matched = res2
    assert matched == "YOLOv8n achieves a 37.3 mAP"

    # 3. Whitespace-normalized match across line breaks
    res3 = find_quote_in_text("COCO val2017 with an inference latency", text)
    assert res3 is not None
    start, end, matched = res3
    assert "\n" in matched or " " in matched
    assert text[start:end] == matched

    # 4. Hallucinated quote (must return None)
    res_fake = find_quote_in_text("RT-DETR achieves 99.9 mAP on ImageNet", text)
    assert res_fake is None

    # 5. Trivial/short quote (must return None)
    res_short = find_quote_in_text("YOLO", text)
    assert res_short is None


def test_evidence_extractor_guards_against_hallucinated_quotes(tmp_path: Path):
    """
    Vertical Slice Guardrail Test:
    Ensures that when LLM returns facts, only facts with authentic, verifiable quotes
    are saved to SQLite. Hallucinated or modified quotes are strictly rejected.
    """
    db = DatabaseManager(db_path=tmp_path / "evidence_test.db")
    from backend.db.repositories import SessionRepository, SourceRepository
    SessionRepository(db).create(session_id="sess_test_extract", goal="Investigate YOLOv8 performance")
    SourceRepository(db).add(
        source_id="src_yolo",
        session_id="sess_test_extract",
        url="https://docs.ultralytics.com/models/yolov8",
        title="Ultralytics YOLOv8 Docs",
        domain="docs.ultralytics.com",
        canonical_key="yolov8"
    )

    sample_chunks = [
        {
            "chunk_id": "chunk_coco_1",
            "source_id": "src_yolo",
            "url": "https://docs.ultralytics.com/models/yolov8",
            "source_title": "Ultralytics YOLOv8 Docs",
            "section": "Performance Metrics",
            "page": 1,
            "char_start": 100,
            "char_end": 350,
            "text": "For instance, the YOLOv8n model achieves a mAP of 37.3 on the COCO dataset and a speed of 0.99 ms on A100 TensorRT."
        }
    ]

    canned = {
        "ExtractedEvidencesSchema": {
            "facts": [
                {
                    "statement": "YOLOv8n achieves 37.3 mAP on COCO dataset with 0.99 ms latency on A100.",
                    "subject": "YOLOv8n",
                    "predicate": "achieves mAP",
                    "metric": "mAP",
                    "value": "37.3",
                    "raw_quote": "YOLOv8n model achieves a mAP of 37.3 on the COCO dataset and a speed of 0.99 ms on A100 TensorRT.",
                    "chunk_id": "chunk_coco_1",
                    "confidence": 0.98
                },
                {
                    "statement": "Fabricated claim not present in text.",
                    "subject": "FakeModel",
                    "predicate": "achieves",
                    "metric": "latency",
                    "value": "0.01 ms",
                    "raw_quote": "This quote does not exist in any chunk anywhere.",
                    "chunk_id": "chunk_coco_1",
                    "confidence": 0.99
                }
            ]
        }
    }

    mock_llm = MockLLMBackend(canned_responses=canned)
    extractor = EvidenceExtractor(llm=mock_llm, db=db)

    results = extractor.extract_from_chunks(
        session_id="sess_test_extract",
        goal="Investigate YOLOv8 performance",
        chunks=sample_chunks
    )

    # Only 1 fact must be accepted (the authentic quote). The fabricated quote must be rejected!
    assert len(results) == 1
    ev = results[0]
    assert ev["subject"] == "YOLOv8n"
    assert ev["value"] == "37.3"
    assert "YOLOv8n model achieves a mAP of 37.3" in ev["exact_quote"]
    assert ev["char_start"] >= 100

    # Verify SQLite persistence in raw_evidences and evidences tables
    raw_evs = extractor.raw_evidence_repo.get_by_session("sess_test_extract")
    assert len(raw_evs) == 1
    assert raw_evs[0]["chunk_id"] == "chunk_coco_1"

    full_evs = extractor.evidence_repo.get_full_evidence_by_session("sess_test_extract")
    assert len(full_evs) == 1
    assert full_evs[0]["url"] == "https://docs.ultralytics.com/models/yolov8"


def test_citation_verifier_appends_table_and_validates_indices():
    evidence_items = [
        {
            "evidence_id": "evi_1",
            "statement": "YOLOv8n achieves 37.3 mAP on COCO.",
            "exact_quote": "YOLOv8n achieves a mAP of 37.3 on COCO.",
            "source_title": "Ultralytics Docs",
            "url": "https://docs.ultralytics.com",
            "section": "Metrics",
            "page": 2
        },
        {
            "evidence_id": "evi_2",
            "statement": "RT-DETR-L achieves 53.0 mAP on COCO.",
            "exact_quote": "RT-DETR-L achieves 53.0 mAP on COCO val2017.",
            "source_title": "Baidu Research",
            "url": "https://arxiv.org/abs/2304.08069",
            "section": "Experiments",
            "page": 4
        }
    ]

    report = (
        "## Object Detection Comparison\n\n"
        "YOLOv8n demonstrates fast inference with 37.3 mAP [E1]. "
        "In contrast, the Transformer-based RT-DETR-L reaches 53.0 mAP [E2]. "
        "Invalid citation check: [E99]."
    )

    augmented, stats = CitationVerifier.verify_and_append_appendix(report, evidence_items)

    assert stats["total_evidence"] == 2
    assert stats["cited_evidence"] == 2
    assert stats["valid_indices"] == [1, 2]
    assert stats["invalid_indices"] == [99]

    # Verify Evidence & Provenance Table formatting
    assert "### Evidence & Provenance Table" in augmented
    assert "| **[E1]** |" in augmented
    assert "| **[E2]** |" in augmented
    assert "https://docs.ultralytics.com" in augmented
    assert "https://arxiv.org/abs/2304.08069" in augmented


def test_api_evidence_endpoints(client, isolated_engine):
    """Verifies GET /api/research/session/{id}/evidence and POST /api/research/session/{id}/extract endpoints."""
    # 1. Create session
    create_res = client.post("/api/research/session", json={"goal": "Compare PointPillars and CenterPoint"})
    session_id = create_res.json()["session_id"]

    # 2. Add sample source, document, and chunk to isolated engine
    isolated_engine.source_repo.add(
        source_id="src_cp",
        session_id=session_id,
        url="https://arxiv.org/abs/2006.11275",
        title="CenterPoint Paper",
        domain="arxiv.org",
        canonical_key="cp"
    )
    isolated_engine.doc_repo.add(
        doc_id="doc_cp",
        source_id="src_cp",
        file_path=None,
        content_hash="hash_cp",
        raw_text="CenterPoint achieves 60.3 mAP and 67.3 NDS on nuScenes benchmark."
    )
    isolated_engine.chunk_repo.add(
        chunk_id="chk_cp_1",
        doc_id="doc_cp",
        text="CenterPoint achieves 60.3 mAP and 67.3 NDS on nuScenes benchmark.",
        section="Results",
        page=5,
        char_start=0,
        char_end=65,
        token_count=12
    )

    # 3. Configure mock LLM to extract this evidence
    canned = {
        "ExtractedEvidencesSchema": {
            "facts": [
                {
                    "statement": "CenterPoint achieves 60.3 mAP on nuScenes.",
                    "subject": "CenterPoint",
                    "predicate": "achieves",
                    "metric": "mAP",
                    "value": "60.3",
                    "raw_quote": "CenterPoint achieves 60.3 mAP",
                    "chunk_id": "chk_cp_1",
                    "confidence": 0.99
                }
            ]
        }
    }
    isolated_engine.llm = MockLLMBackend(canned_responses=canned)

    # 4. Set state to RETRIEVE phase and call POST /extract
    state = isolated_engine.load_state(session_id)
    state.phase = ResearchPhase.RETRIEVE
    state.source_ids.append("src_cp")
    isolated_engine.save_state(state)

    extract_res = client.post(f"/api/research/session/{session_id}/extract")
    assert extract_res.status_code == 200
    extract_data = extract_res.json()
    assert extract_data["evidence_count"] == 1
    assert extract_data["evidence"][0]["subject"] == "CenterPoint"

    # 5. Call GET /evidence
    get_res = client.get(f"/api/research/session/{session_id}/evidence")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["evidence_count"] == 1
    assert data["evidence"][0]["url"] == "https://arxiv.org/abs/2006.11275"
    assert data["evidence"][0]["section"] == "Results"
    assert data["evidence"][0]["page"] == 5


def test_evidence_extractor_rejects_numeric_mismatch_and_subject_mismatch(tmp_path: Path):
    """
    P0-2 Test: Ensures numeric verification and subject grounding guardrails reject fabricated facts:
    1. 'PointPillars achieves 72.9 NDS' with quote 'PointPillars achieves 59.2 NDS' -> REJECTED (NUMERIC_MISMATCH).
    2. 'RT-DETR achieves 53.1 AP on COCO' with quote 'achieves 60.3 mAP' -> REJECTED (SUBJECT_MISMATCH & NUMERIC_MISMATCH).
    """
    from backend.db.repositories import SessionRepository, SourceRepository
    db = DatabaseManager(db_path=tmp_path / "numeric_guard_test.db")
    SessionRepository(db).create(session_id="sess_guard", goal="Compare detection benchmarks")
    SourceRepository(db).add(
        source_id="src_nuscenes",
        session_id="sess_guard",
        url="https://arxiv.org/abs/2006.11275",
        title="nuScenes Benchmark",
        domain="arxiv.org",
        canonical_key="nuscenes"
    )

    chunks = [
        {
            "chunk_id": "chunk_nu_1",
            "source_id": "src_nuscenes",
            "url": "https://arxiv.org/abs/2006.11275",
            "source_title": "nuScenes Benchmark",
            "text": "CenterPoint achieves 60.3 mAP and 67.3 NDS on nuScenes benchmark. Meanwhile, PointPillars achieves 59.2 NDS.",
            "section": "Experiments",
            "page": 5,
            "char_start": 0,
            "char_end": 110
        }
    ]

    canned = {
        "ExtractedEvidencesSchema": {
            "facts": [
                {
                    # Case 1: Fabricated metric number 72.9 (quote actually has 59.2 NDS)
                    "statement": "PointPillars achieves 72.9 NDS, beating CenterPoint.",
                    "subject": "PointPillars",
                    "predicate": "achieves",
                    "metric": "NDS",
                    "value": "72.9",
                    "raw_quote": "PointPillars achieves 59.2 NDS",
                    "chunk_id": "chunk_nu_1",
                    "confidence": 0.99
                },
                {
                    # Case 2: Fabricated subject RT-DETR and fabricated metric 53.1 AP (quote actually has achieves 60.3 mAP)
                    "statement": "RT-DETR achieves 53.1 AP on COCO.",
                    "subject": "RT-DETR",
                    "predicate": "achieves",
                    "metric": "AP",
                    "value": "53.1",
                    "raw_quote": "achieves 60.3 mAP",
                    "chunk_id": "chunk_nu_1",
                    "confidence": 0.95
                },
                {
                    # Case 3: Genuine authentic fact
                    "statement": "CenterPoint achieves 60.3 mAP on nuScenes benchmark.",
                    "subject": "CenterPoint",
                    "predicate": "achieves",
                    "metric": "mAP",
                    "value": "60.3",
                    "raw_quote": "CenterPoint achieves 60.3 mAP",
                    "chunk_id": "chunk_nu_1",
                    "confidence": 0.99
                }
            ]
        }
    }

    mock_llm = MockLLMBackend(canned_responses=canned)
    extractor = EvidenceExtractor(llm=mock_llm, db=db)

    results = extractor.extract_from_chunks(
        session_id="sess_guard",
        goal="Compare detection benchmarks",
        chunks=chunks
    )

    # Both fabricated facts MUST be rejected. Only Case 3 should be accepted!
    assert len(results) == 1
    assert results[0]["subject"] == "CenterPoint"
    assert results[0]["value"] == "60.3"
    assert results[0]["exact_quote"] == "CenterPoint achieves 60.3 mAP"


from backend.tools.web_search import SearchResultItem
from backend.tools.web_fetch import FetchedWebContent


class MockSearch:
    def search(self, query: str, max_results: int = 5):
        return [
            SearchResultItem(
                title="CenterPoint Paper",
                url="https://arxiv.org/abs/2006.11275",
                snippet="CenterPoint achieves 60.3 mAP on nuScenes.",
                query=query,
                rank=1
            )
        ]


class MockFetch:
    def fetch(self, url: str):
        return FetchedWebContent(
            url=url,
            title="CenterPoint 3D Detection",
            text="In our evaluations, CenterPoint achieves 60.3 mAP and 67.3 NDS on nuScenes benchmark.",
            content_hash="hash_cp"
        )


def test_run_week1_end_to_end_extracts_evidence_without_foreign_key_error(tmp_path: Path):
    """
    P0-1 Test: Ensures that run_week1 pipeline with RetrievedChunk objects
    preserves source_id, extracts valid atomic evidence without SQLite foreign key crash,
    and returns evidence_count > 0.
    """
    from backend.core.engine import ResearchEngine
    from backend.core.limits import ResearchLimits
    from backend.retrieval.embeddings import LocalHashEmbeddingBackend

    canned = {
        "ResearchPlanSchema": {
            "goal": "Evaluate CenterPoint 3D detection accuracy",
            "tasks": [
                {"task_id": "t1", "description": "Check nuScenes mAP", "expected_evidence": "60.3 mAP"}
            ],
            "key_hypotheses": ["CenterPoint achieves strong mAP"]
        },
        "GeneratedQueriesSchema": {
            "queries": [
                {"query": "CenterPoint nuScenes mAP benchmark", "query_type": "evidence", "rationale": "benchmark data"}
            ]
        },
        "ExtractedEvidencesSchema": {
            "facts": [
                {
                    "statement": "CenterPoint achieves 60.3 mAP on nuScenes benchmark.",
                    "subject": "CenterPoint",
                    "predicate": "achieves",
                    "metric": "mAP",
                    "value": "60.3",
                    "raw_quote": "CenterPoint achieves 60.3 mAP",
                    "chunk_id": "",
                    "confidence": 0.99
                }
            ]
        }
    }

    db = DatabaseManager(db_path=tmp_path / "e2e_extract_test.db")
    mock_llm = MockLLMBackend(canned_responses=canned)
    engine = ResearchEngine(
        llm=mock_llm,
        db=db,
        limits=ResearchLimits(max_research_steps=1),
        search_tool=MockSearch(),
        fetch_tool=MockFetch(),
        embedding_backend=LocalHashEmbeddingBackend(dimension=64)
    )

    result = engine.run_week1("Evaluate CenterPoint 3D detection accuracy")

    assert result["phase"] == "DONE"
    assert result["status"] == "COMPLETED"
    assert result["evidence_count"] >= 1
    assert "Evidence & Provenance Table" in result["answer"]
    assert "[E1]" in result["answer"]

    # Verify SQLite raw_evidences and evidences were persisted without foreign key errors
    raw_evs = engine.raw_evidence_repo.get_by_session(result["session_id"])
    assert len(raw_evs) >= 1
    assert raw_evs[0]["source_id"].startswith("src_")


def test_extract_core_entities_expanded_terms():
    """
    Verifies improved entity extraction:
    - Drops 'affect', 'using', 'instead', 'tradeoffs', 'consumer', 'speed'
    - Retains 2-letter tech names like 'Go', 'AI'
    """
    from backend.core.coverage import extract_core_entities

    q1 = "How does quantization affect LLM inference speed on consumer GPUs"
    ents1 = extract_core_entities(q1)
    assert "quantization" in ents1
    assert "llm" in ents1
    assert "gpus" in ents1
    assert "affect" not in ents1
    assert "speed" not in ents1
    assert "consumer" not in ents1

    q2 = "What are the tradeoffs of using Rust instead of Go for backend services"
    ents2 = extract_core_entities(q2)
    assert ents2 == ["rust", "go"]


def test_extractor_rejects_evidence_when_source_id_unresolvable(tmp_path: Path):
    """
    Verifies that when an evidence chunk has unknown/missing source_id and cannot be resolved from DB,
    the extractor rejects it rather than falling back to sess_sources[0] (which causes false attribution).
    """
    from backend.db.repositories import SessionRepository, SourceRepository
    db = DatabaseManager(db_path=tmp_path / "source_attrib_test.db")
    SessionRepository(db).create(session_id="sess_attrib", goal="Test attribution")
    # First source belongs to a completely different paper
    SourceRepository(db).add(
        source_id="src_first_unrelated",
        session_id="sess_attrib",
        url="https://arxiv.org/abs/1900.00001",
        title="Unrelated First Paper",
        domain="arxiv.org",
        canonical_key="unrelated"
    )

    chunks = [
        {
            "chunk_id": "chunk_no_source",
            "source_id": "src_unknown",
            "url": "local",
            "text": "CenterPoint achieves 60.3 mAP and 67.3 NDS on nuScenes benchmark.",
            "section": "Results",
            "page": 1,
            "char_start": 0,
            "char_end": 70
        }
    ]

    canned = {
        "ExtractedEvidencesSchema": {
            "facts": [
                {
                    "statement": "CenterPoint achieves 60.3 mAP on nuScenes benchmark.",
                    "subject": "CenterPoint",
                    "predicate": "achieves",
                    "metric": "mAP",
                    "value": "60.3",
                    "raw_quote": "CenterPoint achieves 60.3 mAP",
                    "chunk_id": "chunk_no_source",
                    "confidence": 0.99
                }
            ]
        }
    }

    mock_llm = MockLLMBackend(canned_responses=canned)
    extractor = EvidenceExtractor(llm=mock_llm, db=db)

    results = extractor.extract_from_chunks(
        session_id="sess_attrib",
        goal="Test attribution",
        chunks=chunks
    )

    # Must be rejected because source_id cannot be resolved. NEVER attributed to src_first_unrelated!
    assert len(results) == 0


def test_extractor_rejects_unsupported_comparison_statements(tmp_path: Path):
    """
    Verifies that ungrounded comparative claims (e.g. 'PointPillars is more accurate than CenterPoint')
    are rejected when the verbatim quote only mentions one entity ('PointPillars achieves 59.2 NDS').
    """
    from backend.db.repositories import SessionRepository, SourceRepository
    db = DatabaseManager(db_path=tmp_path / "comparative_test.db")
    SessionRepository(db).create(session_id="sess_comp", goal="Compare PointPillars and CenterPoint")
    SourceRepository(db).add(
        source_id="src_pointpillars",
        session_id="sess_comp",
        url="https://arxiv.org/abs/1812.05784",
        title="PointPillars Paper",
        domain="arxiv.org",
        canonical_key="pointpillars"
    )

    chunks = [
        {
            "chunk_id": "chunk_pp_1",
            "source_id": "src_pointpillars",
            "url": "https://arxiv.org/abs/1812.05784",
            "text": "PointPillars achieves 59.2 NDS on nuScenes val set with fast inference.",
            "section": "Experiments",
            "page": 4,
            "char_start": 0,
            "char_end": 75
        }
    ]

    canned = {
        "ExtractedEvidencesSchema": {
            "facts": [
                {
                    # Comparative statement mentioning CenterPoint without grounding in quote
                    "statement": "PointPillars is more accurate than CenterPoint.",
                    "subject": "PointPillars",
                    "predicate": "more accurate",
                    "metric": "NDS",
                    "value": "59.2",
                    "raw_quote": "PointPillars achieves 59.2 NDS",
                    "chunk_id": "chunk_pp_1",
                    "confidence": 0.95
                },
                {
                    # Grounded authentic statement
                    "statement": "PointPillars achieves 59.2 NDS on nuScenes.",
                    "subject": "PointPillars",
                    "predicate": "achieves",
                    "metric": "NDS",
                    "value": "59.2",
                    "raw_quote": "PointPillars achieves 59.2 NDS",
                    "chunk_id": "chunk_pp_1",
                    "confidence": 0.95
                }
            ]
        }
    }

    mock_llm = MockLLMBackend(canned_responses=canned)
    extractor = EvidenceExtractor(llm=mock_llm, db=db)

    results = extractor.extract_from_chunks(
        session_id="sess_comp",
        goal="Compare PointPillars and CenterPoint",
        chunks=chunks
    )

    # Only the grounded non-comparative statement is accepted
    assert len(results) == 1
    assert results[0]["statement"] == "PointPillars achieves 59.2 NDS on nuScenes."


def test_compute_canonical_key_deduplication():
    """
    Verifies canonical key deduplication across arXiv variants, DOIs, and titles.
    """
    from backend.sources.dedup import compute_canonical_key

    # arXiv normalization
    k1 = compute_canonical_key("https://arxiv.org/abs/2006.11275", "CenterPoint Paper")
    k2 = compute_canonical_key("https://arxiv.org/pdf/2006.11275.pdf", "CenterPoint: Center-based 3D Detection")
    k3 = compute_canonical_key("http://arxiv.org/abs/2006.11275v2", "CenterPoint")
    assert k1 == "arxiv:2006.11275"
    assert k1 == k2 == k3

    # DOI normalization
    d1 = compute_canonical_key("https://doi.org/10.1145/3318464.3389700", "RocksDB paper")
    d2 = compute_canonical_key("http://dx.doi.org/10.1145/3318464.3389700", "RocksDB")
    assert d1 == "doi:10.1145/3318464.3389700"
    assert d1 == d2

    # Title fallback normalization
    t1 = compute_canonical_key("https://example.com/blog/centerpoint-review", "CenterPoint 3D Detection")
    t2 = compute_canonical_key("https://other.com/article/123", "CenterPoint: 3D Detection")
    assert t1 == "title:centerpoint-3d-detection"
    assert t2 == "title:centerpoint-3d-detection"


def test_claims_table_populated_and_retrieved_via_api(tmp_path: Path):
    """
    Verifies that claims table records citation lineage and can be fetched via API.
    """
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.api.research import get_engine
    from backend.retrieval.embeddings import LocalHashEmbeddingBackend
    from backend.core.engine import ResearchEngine
    from backend.core.limits import ResearchLimits

    canned = {
        "ResearchPlanSchema": {
            "goal": "Verify CenterPoint claims",
            "tasks": [{"task_id": "t1", "description": "task", "expected_evidence": "ev"}],
            "key_hypotheses": ["h1"]
        },
        "GeneratedQueriesSchema": {
            "queries": [{"query": "CenterPoint nuScenes", "query_type": "evidence", "rationale": "r"}]
        },
        "ExtractedEvidencesSchema": {
            "facts": [
                {
                    "statement": "CenterPoint achieves 60.3 mAP on nuScenes benchmark.",
                    "subject": "CenterPoint",
                    "predicate": "achieves",
                    "metric": "mAP",
                    "value": "60.3",
                    "raw_quote": "CenterPoint achieves 60.3 mAP",
                    "chunk_id": "",
                    "confidence": 0.99
                }
            ]
        },
        "generate": "CenterPoint achieves 60.3 mAP on nuScenes benchmark [E1]."
    }

    db = DatabaseManager(db_path=tmp_path / "claims_api_test.db")
    mock_llm = MockLLMBackend(canned_responses=canned)
    engine = ResearchEngine(
        llm=mock_llm,
        db=db,
        limits=ResearchLimits(max_research_steps=1),
        search_tool=MockSearch(),
        fetch_tool=MockFetch(),
        embedding_backend=LocalHashEmbeddingBackend(dimension=64)
    )

    app.dependency_overrides[get_engine] = lambda: engine
    client = TestClient(app)

    try:
        run_res = client.post("/api/research/run", json={"goal": "Verify CenterPoint claims"})
        assert run_res.status_code == 200
        sid = run_res.json()["session_id"]

        # Check claims endpoint
        claims_res = client.get(f"/api/research/session/{sid}/claims")
        assert claims_res.status_code == 200
        claims_data = claims_res.json()["claims"]
        assert len(claims_data) >= 1
        assert claims_data[0]["evidence_ids"][0].startswith("evi_")
        assert "CenterPoint" in claims_data[0]["text"]
        assert claims_data[0]["verification"]["citation_index"] == 1
    finally:
        app.dependency_overrides.clear()


def test_session_with_zero_evidence_terminates_partial(tmp_path: Path):
    """
    Verifies that a session with 0 verified evidence extracted transitions to PARTIAL
    with error_message indicating no verified atomic evidence.
    """
    from backend.retrieval.embeddings import LocalHashEmbeddingBackend
    from backend.core.engine import ResearchEngine
    from backend.core.limits import ResearchLimits

    canned = {
        "ResearchPlanSchema": {
            "goal": "Test 0 evidence partial transition",
            "tasks": [{"task_id": "t1", "description": "task", "expected_evidence": "ev"}],
            "key_hypotheses": ["h1"]
        },
        "GeneratedQueriesSchema": {
            "queries": [{"query": "Test query", "query_type": "evidence", "rationale": "r"}]
        },
        # No ExtractedEvidencesSchema -> 0 evidence extracted!
    }

    db = DatabaseManager(db_path=tmp_path / "zero_ev_test.db")
    mock_llm = MockLLMBackend(canned_responses=canned)
    engine = ResearchEngine(
        llm=mock_llm,
        db=db,
        limits=ResearchLimits(max_research_steps=1),
        search_tool=MockSearch(),
        fetch_tool=MockFetch(),
        embedding_backend=LocalHashEmbeddingBackend(dimension=64)
    )

    result = engine.run_week1("Test 0 evidence partial transition")
    assert result["phase"] == "PARTIAL"
    assert result["status"] == "PARTIAL"
    assert result["evidence_count"] == 0
    assert "No verified atomic evidence extracted" in result["error_message"]


def test_expand_quote_to_sentence_ignores_decimal_numbers():
    from backend.evidence.extractor import expand_quote_to_sentence
    text = (
        "Previous methods suffer from high latency. "
        "Our CenterPoint achieves 60.3 mAP and 67.3 NDS on the nuScenes 3D benchmark. "
        "This sets a new state of the art."
    )
    quote = "CenterPoint achieves 60.3 mAP and 67.3 NDS"
    start_idx = text.find(quote)
    end_idx = start_idx + len(quote)
    _, _, expanded = expand_quote_to_sentence(start_idx, end_idx, text)
    assert expanded == "Our CenterPoint achieves 60.3 mAP and 67.3 NDS on the nuScenes 3D benchmark."


def test_verify_atomic_fact_allows_dataset_benchmark_context():
    from backend.evidence.extractor import verify_atomic_fact
    from backend.llm.schemas import AtomicFactItemSchema
    fact = AtomicFactItemSchema(
        statement="CenterPoint achieves 60.3 mAP on nuScenes.",
        subject="CenterPoint",
        predicate="achieves",
        metric="mAP",
        value="60.3",
        raw_quote="CenterPoint achieves 60.3 mAP",
        confidence=0.95
    )
    chunk_dict = {"text": "CenterPoint achieves 60.3 mAP and 67.3 NDS on the nuScenes benchmark."}
    is_valid, reason = verify_atomic_fact(
        fact=fact,
        target_chunk=chunk_dict,
        verified_quote="CenterPoint achieves 60.3 mAP on the nuScenes benchmark.",
        goal_entities=["centerpoint", "nuscenes"]
    )
    assert is_valid is True
    assert reason == "VERIFIED"


def test_canonical_key_preserves_github_subpaths_and_supports_arxiv_html():
    from backend.sources.dedup import compute_canonical_key

    # arXiv HTML support
    k_html = compute_canonical_key("https://arxiv.org/html/2304.08069v1", "RT-DETR Paper")
    k_abs = compute_canonical_key("https://arxiv.org/abs/2304.08069", "RT-DETR Paper")
    assert k_html == "arxiv:2304.08069"
    assert k_html == k_abs

    # GitHub subpaths must NOT collide based on common repo title
    gh_root = compute_canonical_key(
        "https://github.com/open-mmlab/OpenPCDet",
        "GitHub - open-mmlab/OpenPCDet: An open-source 3D detection codebase"
    )
    gh_docs = compute_canonical_key(
        "https://github.com/open-mmlab/OpenPCDet/blob/master/docs/getting_started.md",
        "GitHub - open-mmlab/OpenPCDet: An open-source 3D detection codebase"
    )
    assert gh_root != gh_docs
    assert gh_root == "url:github.com/open-mmlab/openpcdet"
    assert gh_docs == "url:github.com/open-mmlab/openpcdet/blob/master/docs/getting_started.md"


def test_uncited_numeric_report_creates_unsupported_claim_and_forces_partial(tmp_path: Path):
    from backend.core.engine import ResearchEngine
    from backend.core.limits import ResearchLimits
    from backend.retrieval.embeddings import LocalHashEmbeddingBackend

    canned = {
        "ResearchPlanSchema": {
            "goal": "Evaluate CenterPoint detection",
            "tasks": [{"task_id": "t1", "description": "nuScenes test", "expected_evidence": "60.3"}],
            "key_hypotheses": ["CenterPoint"]
        },
        "GeneratedQueriesSchema": {
            "queries": [{"query": "CenterPoint", "query_type": "evidence", "rationale": "r"}]
        },
        "ExtractedEvidencesSchema": {
            "facts": [
                {
                    "statement": "CenterPoint achieves 60.3 mAP.",
                    "subject": "CenterPoint",
                    "predicate": "achieves",
                    "metric": "mAP",
                    "value": "60.3",
                    "raw_quote": "CenterPoint achieves 60.3 mAP",
                    "chunk_id": "",
                    "confidence": 0.99
                }
            ]
        },
        # The LLM answers with an uncited fabricated numeric claim!
        "generate": (
            "CenterPoint achieves 60.3 mAP on nuScenes [E1]. "
            "However, PointPillars is best with 80.5 NDS without citations."
        )
    }

    db = DatabaseManager(db_path=tmp_path / "uncited_num_test.db")
    mock_llm = MockLLMBackend(canned_responses=canned)
    engine = ResearchEngine(
        llm=mock_llm,
        db=db,
        limits=ResearchLimits(max_research_steps=1),
        search_tool=MockSearch(),
        fetch_tool=MockFetch(),
        embedding_backend=LocalHashEmbeddingBackend(dimension=64)
    )

    result = engine.run_week1("Evaluate CenterPoint detection")

    # Because of the uncited numeric claim 'PointPillars is best with 80.5 NDS without citations',
    # the session MUST transition to PARTIAL!
    assert result["phase"] == "PARTIAL"
    assert result["status"] == "PARTIAL"
    assert "Report contains unsupported numeric claims without citations" in result["error_message"]

    # Verify claim lineage in DB
    claims = engine.claim_repo.get_by_session(result["session_id"])
    supported = [c for c in claims if c["status"] == "CITED"]
    unsupported = [c for c in claims if c["status"] == "UNSUPPORTED"]

    assert len(supported) >= 1
    assert supported[0]["verification"]["verified"] is True
    assert supported[0]["evidence_ids"][0].startswith("evi_")

    assert len(unsupported) >= 1
    assert unsupported[0]["verification"]["verified"] is False
    assert "PointPillars is best with 80.5 NDS" in unsupported[0]["text"]


def test_evidence_based_entity_coverage_triggers_partial_when_one_entity_lacks_evidence(tmp_path: Path):
    from backend.core.engine import ResearchEngine
    from backend.core.limits import ResearchLimits
    from backend.retrieval.embeddings import LocalHashEmbeddingBackend

    canned = {
        "ResearchPlanSchema": {
            "goal": "Compare CenterPoint and PointPillars",
            "tasks": [{"task_id": "t1", "description": "compare", "expected_evidence": "ev"}],
            "key_hypotheses": ["h1"]
        },
        "GeneratedQueriesSchema": {
            "queries": [{"query": "CenterPoint vs PointPillars", "query_type": "evidence", "rationale": "r"}]
        },
        # Evidence only exists for CenterPoint, NONE for PointPillars!
        "ExtractedEvidencesSchema": {
            "facts": [
                {
                    "statement": "CenterPoint achieves 60.3 mAP on nuScenes.",
                    "subject": "CenterPoint",
                    "predicate": "achieves",
                    "metric": "mAP",
                    "value": "60.3",
                    "raw_quote": "CenterPoint achieves 60.3 mAP",
                    "chunk_id": "",
                    "confidence": 0.99
                }
            ]
        },
        "generate": "CenterPoint achieves 60.3 mAP on nuScenes benchmark [E1]."
    }

    db = DatabaseManager(db_path=tmp_path / "entity_ev_cov_test.db")
    mock_llm = MockLLMBackend(canned_responses=canned)
    engine = ResearchEngine(
        llm=mock_llm,
        db=db,
        limits=ResearchLimits(max_research_steps=1),
        search_tool=MockSearch(),
        fetch_tool=MockFetch(),
        embedding_backend=LocalHashEmbeddingBackend(dimension=64)
    )

    result = engine.run_week1("Compare CenterPoint and PointPillars")
    # Must transition to PARTIAL because PointPillars lacks verified atomic evidence!
    assert result["phase"] == "PARTIAL"
    assert result["status"] == "PARTIAL"
    assert "Missing evidence for: pointpillars" in result["error_message"]


def test_cited_sentence_with_mismatched_numbers_triggers_numeric_mismatch_and_partial(tmp_path: Path):
    """
    Scenario 4: Report cites [E1] but contains fabricated metric (95.0 NDS)
    not found in the verbatim quote (which has 67.3 NDS).
    Must mark claim as NUMERIC_MISMATCH, verified=False, and force session to PARTIAL!
    """
    from backend.core.engine import ResearchEngine
    from backend.core.limits import ResearchLimits
    from backend.retrieval.embeddings import LocalHashEmbeddingBackend

    canned = {
        "ResearchPlanSchema": {
            "goal": "Evaluate CenterPoint detection",
            "tasks": [{"task_id": "t1", "description": "nuScenes test", "expected_evidence": "NDS"}],
            "key_hypotheses": ["detection"]
        },
        "GeneratedQueriesSchema": {
            "queries": [{"query": "CenterPoint nuScenes", "query_type": "evidence", "rationale": "r"}]
        },
        "ExtractedEvidencesSchema": {
            "facts": [
                {
                    "statement": "CenterPoint achieves 60.3 mAP and 67.3 NDS on nuScenes.",
                    "subject": "CenterPoint",
                    "predicate": "achieves",
                    "metric": "NDS",
                    "value": "67.3",
                    "raw_quote": "CenterPoint achieves 60.3 mAP and 67.3 NDS on nuScenes",
                    "chunk_id": "",
                    "confidence": 0.99
                }
            ]
        },
        # LLM cites [E1] but fabricates a 95.0 NDS metric!
        "generate": "CenterPoint reaches 95.0 NDS on nuScenes benchmark [E1]."
    }

    db = DatabaseManager(db_path=tmp_path / "scenario_4_test.db")
    mock_llm = MockLLMBackend(canned_responses=canned)
    engine = ResearchEngine(
        llm=mock_llm,
        db=db,
        limits=ResearchLimits(max_research_steps=1),
        search_tool=MockSearch(),
        fetch_tool=MockFetch(),
        embedding_backend=LocalHashEmbeddingBackend(dimension=64)
    )

    result = engine.run_week1("Evaluate CenterPoint detection")

    assert result["phase"] == "PARTIAL"
    assert result["status"] == "PARTIAL"
    assert "Report contains numeric claims mismatched with cited evidence" in result["error_message"]

    claims = engine.claim_repo.get_by_session(result["session_id"])
    mismatch_claims = [c for c in claims if c["status"] == "NUMERIC_MISMATCH"]
    assert len(mismatch_claims) == 1
    assert mismatch_claims[0]["verification"]["verified"] is False
    assert mismatch_claims[0]["verification"]["numeric_match"] is False
    assert "95.0" in mismatch_claims[0]["verification"]["mismatched_numbers"]


def test_sentence_citing_multiple_evidences_deduplicates_to_single_claim(tmp_path: Path):
    """
    Ensures that when a sentence cites multiple evidences (e.g. [E1] and [E2]),
    it is stored as ONE single claim record with evidence_ids=[E1, E2] rather than duplicating.
    """
    from backend.core.engine import ResearchEngine
    from backend.core.limits import ResearchLimits
    from backend.retrieval.embeddings import LocalHashEmbeddingBackend

    canned = {
        "ResearchPlanSchema": {
            "goal": "Evaluate CenterPoint detection",
            "tasks": [{"task_id": "t1", "description": "task", "expected_evidence": "ev"}],
            "key_hypotheses": ["h1"]
        },
        "GeneratedQueriesSchema": {
            "queries": [{"query": "CenterPoint", "query_type": "evidence", "rationale": "r"}]
        },
        "ExtractedEvidencesSchema": {
            "facts": [
                {
                    "statement": "CenterPoint achieves 60.3 mAP.",
                    "subject": "CenterPoint",
                    "predicate": "achieves",
                    "metric": "mAP",
                    "value": "60.3",
                    "raw_quote": "CenterPoint achieves 60.3 mAP",
                    "chunk_id": "",
                    "confidence": 0.99
                },
                {
                    "statement": "CenterPoint achieves 67.3 NDS.",
                    "subject": "CenterPoint",
                    "predicate": "achieves",
                    "metric": "NDS",
                    "value": "67.3",
                    "raw_quote": "CenterPoint achieves 60.3 mAP and 67.3 NDS on nuScenes benchmark.",
                    "chunk_id": "",
                    "confidence": 0.99
                }
            ]
        },
        # Sentence citing both [E1] and [E2]
        "generate": "CenterPoint achieves 60.3 mAP [E1] and 67.3 NDS on the benchmark [E2]."
    }

    db = DatabaseManager(db_path=tmp_path / "dedup_claims_test.db")
    mock_llm = MockLLMBackend(canned_responses=canned)
    engine = ResearchEngine(
        llm=mock_llm,
        db=db,
        limits=ResearchLimits(max_research_steps=1),
        search_tool=MockSearch(),
        fetch_tool=MockFetch(),
        embedding_backend=LocalHashEmbeddingBackend(dimension=64)
    )

    result = engine.run_week1("Evaluate CenterPoint detection")
    assert result["phase"] == "DONE"

    claims = engine.claim_repo.get_by_session(result["session_id"])
    assert len(claims) == 1
    assert claims[0]["status"] == "CITED"
    assert len(claims[0]["evidence_ids"]) == 2
    assert claims[0]["verification"]["citation_indices"] == [1, 2]


def test_year_and_small_integer_counts_do_not_trigger_uncited_claims(tmp_path: Path):
    """
    Ensures that calendar years (e.g. 2019) and small count integers without units (e.g. '2 models', '3 stages')
    do NOT trigger uncited numeric claim penalties or false PARTIAL transitions.
    """
    from backend.core.engine import ResearchEngine
    from backend.core.limits import ResearchLimits
    from backend.retrieval.embeddings import LocalHashEmbeddingBackend

    canned = {
        "ResearchPlanSchema": {
            "goal": "Evaluate CenterPoint detection",
            "tasks": [{"task_id": "t1", "description": "task", "expected_evidence": "ev"}],
            "key_hypotheses": ["h1"]
        },
        "GeneratedQueriesSchema": {
            "queries": [{"query": "CenterPoint", "query_type": "evidence", "rationale": "r"}]
        },
        "ExtractedEvidencesSchema": {
            "facts": [
                {
                    "statement": "CenterPoint achieves 60.3 mAP.",
                    "subject": "CenterPoint",
                    "predicate": "achieves",
                    "metric": "mAP",
                    "value": "60.3",
                    "raw_quote": "CenterPoint achieves 60.3 mAP",
                    "chunk_id": "",
                    "confidence": 0.99
                }
            ]
        },
        # Sentence contains year 2019 and small counts 2 and 3 without metric units
        "generate": "In 2019, researchers evaluated 2 models across 3 experimental phases. CenterPoint achieves 60.3 mAP [E1]."
    }

    db = DatabaseManager(db_path=tmp_path / "year_filter_test.db")
    mock_llm = MockLLMBackend(canned_responses=canned)
    engine = ResearchEngine(
        llm=mock_llm,
        db=db,
        limits=ResearchLimits(max_research_steps=1),
        search_tool=MockSearch(),
        fetch_tool=MockFetch(),
        embedding_backend=LocalHashEmbeddingBackend(dimension=64)
    )

    result = engine.run_week1("Evaluate CenterPoint detection")
    assert result["phase"] == "DONE"
    assert result["status"] == "COMPLETED"

    claims = engine.claim_repo.get_by_session(result["session_id"])
    unsupported = [c for c in claims if c["status"] == "UNSUPPORTED"]
    assert len(unsupported) == 0


def test_cross_domain_benchmarks_and_syntactic_context_extraction():
    from backend.core.coverage import extract_context_entities, extract_core_entities
    from backend.evidence.extractor import verify_atomic_fact
    from backend.llm.schemas import AtomicFactItemSchema

    # 1. Database benchmark extraction
    g1 = "Compare PostgreSQL and MySQL on TPC-C"
    ctx1 = extract_context_entities(g1)
    assert "tpc-c" in ctx1 or "tpcc" in ctx1

    # 2. LLM benchmark extraction
    g2 = "Evaluate Llama 3 vs Qwen 2 on MMLU and GSM8K"
    ctx2 = extract_context_entities(g2)
    assert "mmlu" in ctx2
    assert "gsm8k" in ctx2

    # 3. Systems benchmark extraction with 'using'
    g3 = "Compare RocksDB and LevelDB using db_bench"
    ctx3 = extract_context_entities(g3)
    assert "db_bench" in ctx3

    # 4. verify_atomic_fact allows context entity from goal across domains
    fact = AtomicFactItemSchema(
        statement="PostgreSQL achieves 12500 tpmC on TPC-C benchmark.",
        subject="PostgreSQL",
        predicate="achieves",
        metric="tpmC",
        value="12500",
        raw_quote="PostgreSQL achieves 12500 tpmC on TPC-C benchmark",
        confidence=0.98
    )
    is_valid, reason = verify_atomic_fact(
        fact=fact,
        target_chunk={"text": "In our evaluation, PostgreSQL achieves 12500 tpmC on TPC-C benchmark."},
        verified_quote="PostgreSQL achieves 12500 tpmC on TPC-C benchmark",
        goal="Compare PostgreSQL and MySQL on TPC-C",
        goal_entities=extract_core_entities("Compare PostgreSQL and MySQL on TPC-C")
    )
    assert is_valid is True
    assert reason == "VERIFIED"


def test_source_credibility_scoring_plan_14():
    from backend.sources.credibility import score_source_credibility

    # Academic primary paper (arXiv)
    arxiv_res = score_source_credibility(
        url="https://arxiv.org/abs/2304.08069",
        title="DETRs Beat YOLOs on Real-time Object Detection",
        text="We report extensive benchmark results and experiments on COCO val2017."
    )
    assert arxiv_res["tier"] == "TIER_1_ACADEMIC"
    assert arxiv_res["authority"] == 1.0
    assert arxiv_res["primary_source"] == 1.0
    assert arxiv_res["score"] >= 0.85

    # Official open source repository
    gh_res = score_source_credibility(
        url="https://github.com/open-mmlab/OpenPCDet",
        title="OpenPCDet Toolbox",
        text="OpenPCDet is an open source 3D object detection codebase with reproducible benchmarks."
    )
    assert gh_res["tier"] == "TIER_2_OFFICIAL"
    assert gh_res["reproducibility"] == 1.0
    assert gh_res["score"] >= 0.80

    # Third-party community blog
    blog_res = score_source_credibility(
        url="https://medium.com/@random_user/my-thoughts-on-yolo",
        title="My Thoughts on YOLO",
        text="In my opinion, YOLO is cool."
    )
    assert blog_res["tier"] == "TIER_3_COMMUNITY_BLOG"
    assert blog_res["authority"] == 0.50
    assert blog_res["score"] <= 0.60



