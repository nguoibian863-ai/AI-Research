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


def test_run_week1_end_to_end_extracts_evidence_without_foreign_key_error(tmp_path: Path):
    """
    P0-1 Test: Ensures that run_week1 pipeline with RetrievedChunk objects
    preserves source_id, extracts valid atomic evidence without SQLite foreign key crash,
    and returns evidence_count > 0.
    """
    from backend.core.engine import ResearchEngine
    from backend.core.limits import ResearchLimits
    from backend.retrieval.embeddings import LocalHashEmbeddingBackend
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

