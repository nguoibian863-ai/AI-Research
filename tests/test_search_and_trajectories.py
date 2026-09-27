import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock

import pytest
import httpx

from backend.tools.search_providers import (
    SearchResultItem,
    ArxivSearchProvider,
    OpenAlexSearchProvider,
    CompositeSearchProvider,
    reconstruct_openalex_abstract,
)
from backend.tools.web_search import WebSearchTool
from backend.evidence.extractor import EvidenceExtractor, EXTRACTOR_PROMPT_VERSION
from backend.llm.schemas import ExtractedEvidencesSchema, AtomicFactItemSchema, GeneratedQueriesSchema, SearchQueryItemSchema
from backend.core.state import ResearchState, ResearchPhase, ResearchPlan, PlanTask


SAMPLE_ARXIV_XML = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2304.08069v1</id>
    <title>DETRs Beat YOLOs on Real-time Object Detection</title>
    <summary>Real-time object detector with Transformer architecture achieves superior speed and accuracy on COCO.</summary>
    <published>2023-04-17T09:00:00Z</published>
    <arxiv:doi>10.1109/CVPR.2024.12345</arxiv:doi>
  </entry>
</feed>
"""

SAMPLE_OPENALEX_JSON = {
    "results": [
        {
            "id": "https://openalex.org/W3123456789",
            "title": "A Comprehensive Survey on Transaction Processing Performance Council Benchmarks",
            "doi": "https://doi.org/10.1145/123456.7890",
            "publication_year": 2023,
            "cited_by_count": 42,
            "open_access": {
                "is_oa": True,
                "oa_url": "https://arxiv.org/pdf/2304.08069.pdf"
            },
            "primary_location": {
                "landing_page_url": "https://dl.acm.org/doi/10.1145/123456.7890",
                "source": {
                    "display_name": "ACM Computing Surveys"
                }
            },
            "abstract_inverted_index": {
                "TPC-C": [0],
                "evaluates": [1],
                "database": [2],
                "transaction": [3],
                "throughput": [4],
            }
        }
    ]
}


def test_arxiv_search_provider_offline():
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_ARXIV_XML
    mock_client.get.return_value = mock_resp

    provider = ArxivSearchProvider(timeout=5, client=mock_client)
    items = provider.search("DETR YOLO COCO", max_results=3)

    assert len(items) == 1
    item = items[0]
    assert item.title == "DETRs Beat YOLOs on Real-time Object Detection"
    # Full-text HTML URL preferred over /abs/ for rich benchmark and experiment extraction
    assert item.url == "https://arxiv.org/html/2304.08069v1"
    assert item.arxiv_id == "2304.08069v1"
    assert item.source_type == "paper"
    assert item.doi == "10.1109/CVPR.2024.12345"
    assert "Transformer architecture" in item.snippet


def test_arxiv_search_provider_error_handling():
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.side_effect = httpx.ConnectTimeout("Network timeout")

    provider = ArxivSearchProvider(timeout=5, client=mock_client)
    items = provider.search("Error query", max_results=3)
    assert items == []


def test_openalex_search_provider_prioritizes_oa_url():
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = SAMPLE_OPENALEX_JSON
    mock_client.get.return_value = mock_resp

    provider = OpenAlexSearchProvider(timeout=5, client=mock_client)
    items = provider.search("TPC-C database benchmark", max_results=3)

    assert len(items) == 1
    item = items[0]
    assert item.title == "A Comprehensive Survey on Transaction Processing Performance Council Benchmarks"
    # Prioritizes open-access / PDF over paywalled publisher landing page (Issue 3)
    assert item.url == "https://arxiv.org/pdf/2304.08069.pdf"
    assert item.source_type == "academic"
    assert item.venue == "ACM Computing Surveys"
    assert item.citation_count == 42
    assert item.published_at == "2023"
    assert item.snippet == "TPC-C evaluates database transaction throughput"


def test_openalex_search_provider_rate_limit_graceful():
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = json.dumps({"error": "Rate limit exceeded"})
    mock_client.get.return_value = mock_resp

    provider = OpenAlexSearchProvider(timeout=5, client=mock_client)
    items = provider.search("Rate limited query", max_results=3)
    # Must not raise exception, gracefully returns empty list
    assert items == []


def test_composite_search_provider_round_robin_interleaving_and_quota():
    """Verifies that all providers are invoked and interleaved (Issue 1 fix)."""
    calls = {"arxiv": 0, "openalex": 0, "ddg": 0}

    class FakeArxiv:
        name = "arxiv"
        def search(self, query, max_results=8):
            calls["arxiv"] += 1
            return [
                SearchResultItem(title=f"Arxiv Paper {i}", url=f"https://arxiv.org/html/2301.0000{i}", snippet="Paper", source_type="paper")
                for i in range(1, 9)
            ]

    class FakeOpenAlex:
        name = "openalex"
        def search(self, query, max_results=8):
            calls["openalex"] += 1
            return [
                SearchResultItem(title=f"OpenAlex Paper {i}", url=f"https://doi.org/10.1000/{i}", snippet="Academic", source_type="academic")
                for i in range(1, 9)
            ]

    class FakeDDG:
        name = "duckduckgo"
        def search(self, query, max_results=8):
            calls["ddg"] += 1
            return [
                SearchResultItem(title=f"Web Doc {i}", url=f"https://ultralytics.com/docs/{i}", snippet="Doc", source_type="web")
                for i in range(1, 9)
            ]

    composite = CompositeSearchProvider(providers=[FakeArxiv(), FakeOpenAlex(), FakeDDG()], max_results=6)
    results = composite.search("Object detection")

    # ALL providers must be called (Issue 1: previously only arXiv was called)
    assert calls["arxiv"] >= 1
    assert calls["openalex"] >= 1
    assert calls["ddg"] >= 1

    # Interleaved round-robin order
    assert len(results) == 6
    assert results[0].source_type == "paper"      # Arxiv [0]
    assert results[1].source_type == "academic"   # OpenAlex [0]
    assert results[2].source_type == "web"        # DDG [0]
    assert results[3].source_type == "paper"      # Arxiv [1]
    assert results[4].source_type == "academic"   # OpenAlex [1]
    assert results[5].source_type == "web"        # DDG [1]


def test_trajectory_logging_query_generation_and_evidence_extraction(isolated_engine, tmp_path):
    engine = isolated_engine
    state = ResearchState(
        session_id="sess_traj_test",
        goal="Compare YOLOv8 and RT-DETR accuracy and latency on COCO",
        phase=ResearchPhase.PLAN
    )
    state.plan = ResearchPlan(
        goal=state.goal,
        tasks=[PlanTask(task_id="t1", description="Evaluate YOLOv8 accuracy", expected_evidence="mAP")],
        key_hypotheses=[]
    )
    engine.session_repo.create(state.session_id, state.goal, phase="PLAN", status="PENDING")

    # 1. Run Search Phase with LLM query generation -> query_origin = "llm"
    mock_q_res = MagicMock()
    mock_q_res.parsed = GeneratedQueriesSchema(queries=[SearchQueryItemSchema(query="YOLOv8 RT-DETR benchmark")])
    mock_q_res.total_tokens = 50
    mock_q_res.calls_made = 1
    engine.llm.structured_generate = MagicMock(return_value=mock_q_res)

    engine.run_search_phase(state)

    query_trajs = engine.trajectory_repo.list_by_task_type("query_generation", partition="raw")
    assert len(query_trajs) >= 1
    qt = query_trajs[0]
    payload = json.loads(qt["payload_json"])
    assert payload["goal"] == state.goal
    assert payload["query_origin"] == "llm"  # Issue 4 fix
    assert qt["verified"] == 0               # Issue 4 fix: verified=False for raw queries

    # 1b. Run search phase with custom queries (open_questions) -> query_origin = "open_questions"
    state.phase = ResearchPhase.EVALUATE
    engine.run_search_phase(state, custom_queries=["RT-DETR latency T4"])
    oq_trajs = [t for t in engine.trajectory_repo.list_by_task_type("query_generation", partition="raw")
                if json.loads(t["payload_json"]).get("query_origin") == "open_questions"]
    assert len(oq_trajs) >= 1

    # 2. Run Evidence Extraction -> logs evidence_extraction trajectory for each candidate fact
    chunk = {
        "chunk_id": "chk_doc_test_001",
        "source_id": state.source_ids[0] if state.source_ids else "src_test_001",
        "text": "RT-DETR achieves 53.1 mAP on COCO val2017 with 74 FPS on T4 GPU.",
        "section": "Experiments",
        "page": 1,
        "url": "https://arxiv.org/abs/2304.08069"
    }

    # Setup mock LLM response proposing 1 valid fact and 1 hallucinated fact
    valid_fact = AtomicFactItemSchema(
        statement="RT-DETR achieves 53.1 mAP on COCO val2017.",
        subject="RT-DETR",
        predicate="achieves",
        metric="mAP",
        value="53.1",
        raw_quote="RT-DETR achieves 53.1 mAP on COCO val2017 with 74 FPS on T4 GPU."
    )
    ungrounded_fact = AtomicFactItemSchema(
        statement="YOLOv8 achieves 99.9 mAP on ImageNet.",
        subject="YOLOv8",
        predicate="achieves",
        metric="mAP",
        value="99.9",
        raw_quote="YOLOv8 achieves 99.9 mAP on ImageNet."  # Not in chunk!
    )

    mock_llm_res = MagicMock()
    mock_llm_res.parsed = ExtractedEvidencesSchema(facts=[valid_fact, ungrounded_fact])
    mock_llm_res.calls_made = 1
    mock_llm_res.total_tokens = 250
    engine.evidence_extractor.llm.structured_generate = MagicMock(return_value=mock_llm_res)

    extracted = engine.evidence_extractor.extract_from_chunks(
        session_id=state.session_id,
        goal=state.goal,
        chunks=[chunk],
        max_evidence=5
    )

    # Valid fact was verified and saved
    assert len(extracted) == 1
    assert extracted[0]["value"] == "53.1"

    # Verify trajectory table recorded BOTH facts with prompt and prompt_version (Issue 5 fix)
    ext_trajs = engine.trajectory_repo.list_by_task_type("evidence_extraction", partition="raw")
    assert len(ext_trajs) >= 2

    for t in ext_trajs:
        p = json.loads(t["payload_json"])
        assert p["prompt_version"] == EXTRACTOR_PROMPT_VERSION  # Issue 5
        assert p["prompt"] is not None and "Goal:" in p["prompt"]  # Issue 5

    verdicts = [json.loads(t["payload_json"])["verdict"] for t in ext_trajs]
    assert "VERIFIED" in verdicts
    assert "QUOTE_NOT_FOUND" in verdicts

    # 3. Test JSONL export with llm_only_queries filter per Plan 43.1 / 42.4 (Issue 4)
    counts = engine.trajectory_repo.export_partition_jsonl(
        partition="raw",
        output_dir=tmp_path / "trajectories",
        llm_only_queries=True
    )
    # The open_questions query was filtered out, keeping only the LLM-generated query sample!
    assert counts.get("query_generation", 0) == 1
    assert counts.get("evidence_extraction", 0) >= 2

    q_file = tmp_path / "trajectories" / "raw" / "query_generation.jsonl"
    q_lines = [json.loads(line) for line in q_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(q_lines) == 1
    assert q_lines[0]["payload"]["query_origin"] == "llm"


def test_composite_search_fills_quota_when_providers_fail_or_empty():
    """Verifies that composite search compensates when some providers fail (Issue 4 & 5 fix)."""
    class GoodProvider:
        name = "arxiv"
        def search(self, query, max_results=8):
            # Respect requested max_results: if caller requests 3, return 3; if 6, return 6
            return [
                SearchResultItem(title=f"Arxiv {i}", url=f"https://arxiv.org/html/2301.000{i}", snippet="Paper", source_type="paper")
                for i in range(1, max_results + 1)
            ]

    class FailingProvider:
        name = "openalex"
        def search(self, query, max_results=8):
            raise RuntimeError("API timeout")

    class EmptyProvider:
        name = "duckduckgo"
        def search(self, query, max_results=8):
            return []  # e.g. Rate-limited

    composite = CompositeSearchProvider(providers=[GoodProvider(), FailingProvider(), EmptyProvider()], max_results=6)
    results = composite.search("Object detection")

    # Even though 2 of 3 providers failed/empty, the working provider fills the quota (6 results)
    assert len(results) == 6
    assert all(r.source_type == "paper" for r in results)


def test_extract_early_stopping_on_comparative_subjects_with_dataset_goal(isolated_engine):
    """Verifies early stop triggers when compared subjects are covered, ignoring dataset entity (Issue 2 fix)."""
    engine = isolated_engine
    goal = "Compare YOLOv8 and RT-DETR accuracy and latency on COCO"
    session_id = "sess_early_stop_test"

    engine.session_repo.create(session_id, goal, phase="EXTRACT", status="PENDING")
    engine.source_repo.add(
        source_id="s1", session_id=session_id, url="https://arxiv.org/abs/2304.08069",
        title="Paper", source_type="paper", domain="arxiv.org", canonical_key="arxiv:2304.08069"
    )

    chunk1 = {"chunk_id": "c1", "text": "YOLOv8 achieves 53.9 mAP on COCO val2017. YOLOv8 processes frames at 280 FPS on T4 GPU.", "source_id": "s1"}
    chunk2 = {"chunk_id": "c2", "text": "RT-DETR-R50 achieves 53.1 mAP with 108 FPS.", "source_id": "s1"}
    chunk3 = {"chunk_id": "c3", "text": "Another document about detectors.", "source_id": "s1"}

    fact1 = AtomicFactItemSchema(
        statement="YOLOv8 achieves 53.9 mAP on COCO val2017.",
        subject="YOLOv8",
        predicate="achieves",
        metric="mAP",
        value="53.9",
        raw_quote="YOLOv8 achieves 53.9 mAP on COCO val2017."
    )
    fact2 = AtomicFactItemSchema(
        statement="YOLOv8 processes frames at 280 FPS on T4 GPU.",
        subject="YOLOv8",
        predicate="processes",
        metric="FPS",
        value="280",
        raw_quote="YOLOv8 processes frames at 280 FPS on T4 GPU."
    )
    fact3 = AtomicFactItemSchema(
        statement="RT-DETR-R50 achieves 53.1 mAP with 108 FPS.",
        subject="RT-DETR-R50",
        predicate="achieves",
        metric="mAP",
        value="53.1",
        raw_quote="RT-DETR-R50 achieves 53.1 mAP with 108 FPS."
    )

    # Chunk 1 returns 2 facts (YOLOv8), Chunk 2 returns 1 fact (RT-DETR-R50)
    mock_res_c1 = MagicMock()
    mock_res_c1.parsed = ExtractedEvidencesSchema(facts=[fact1, fact2])
    mock_res_c1.calls_made = 1
    mock_res_c1.total_tokens = 100

    mock_res_c2 = MagicMock()
    mock_res_c2.parsed = ExtractedEvidencesSchema(facts=[fact3])
    mock_res_c2.calls_made = 1
    mock_res_c2.total_tokens = 100

    # Chunk 3 should never be called
    mock_res_c3 = MagicMock()
    mock_res_c3.parsed = ExtractedEvidencesSchema(facts=[])

    engine.evidence_extractor.llm.structured_generate = MagicMock(side_effect=[mock_res_c1, mock_res_c2, mock_res_c3])

    extracted = engine.evidence_extractor.extract_from_chunks(
        session_id="sess_early_stop_test",
        goal=goal,
        chunks=[chunk1, chunk2, chunk3],
        max_evidence=10
    )

    # 3 facts collected, both yolov8 and rt-detr covered -> stops early!
    assert len(extracted) == 3
    # LLM was only called 2 times, chunk 3 was skipped!
    assert engine.evidence_extractor.llm.structured_generate.call_count == 2


def test_extract_write_reservation_dynamic(isolated_engine):
    """Verifies that extraction stops early when remaining time is insufficient for WRITE (Issue 1 fix)."""
    engine = isolated_engine
    goal = "Compare PostgreSQL and MySQL on TPC-C"

    chunk1 = {"chunk_id": "c1", "text": "PostgreSQL achieves 1000 TPS on TPC-C benchmark.", "source_id": "s1"}
    chunk2 = {"chunk_id": "c2", "text": "MySQL achieves 950 TPS on TPC-C benchmark.", "source_id": "s1"}

    # Mock budget tracker with explicit assert_can_call_llm to avoid MagicMock AttributeError masking
    mock_tracker = MagicMock()
    mock_tracker.can_call_llm.return_value = True
    mock_tracker.assert_can_call_llm = MagicMock()
    mock_tracker.record_llm_call = MagicMock()
    mock_tracker.limits.max_runtime_seconds = 600
    mock_tracker.limits.write_reserved_seconds = 90.0

    # Branch A: remaining time is 130s (600 - 470).
    # On old code: threshold was 120s (90 + 30), so 130s did NOT stop -> would call LLM.
    # On new code: threshold is 170s (90 + 80), so 130s stops cleanly before chunk 1.
    mock_tracker.elapsed_seconds = 470.0
    engine.evidence_extractor.llm.structured_generate = MagicMock()

    extracted = engine.evidence_extractor.extract_from_chunks(
        session_id="sess_reserve_test",
        goal=goal,
        chunks=[chunk1, chunk2],
        budget_tracker=mock_tracker,
        max_evidence=5
    )

    # Stopped before even starting chunk 1!
    assert len(extracted) == 0
    assert engine.evidence_extractor.llm.structured_generate.call_count == 0

    # Branch B: remaining time is 220s (600 - 380) > 170s threshold.
    # LLM MUST be called when time is sufficient!
    mock_tracker.elapsed_seconds = 380.0
    mock_res = MagicMock()
    mock_res.parsed.evidences = []
    mock_res.total_tokens = 50
    mock_res.calls_made = 1
    engine.evidence_extractor.llm.structured_generate = MagicMock(return_value=mock_res)

    engine.evidence_extractor.extract_from_chunks(
        session_id="sess_reserve_test",
        goal=goal,
        chunks=[chunk1, chunk2],
        budget_tracker=mock_tracker,
        max_evidence=5
    )
    assert engine.evidence_extractor.llm.structured_generate.call_count > 0


def test_subject_matches_entity_strict_semantics():
    """Verifies subject matching prevents family names (DETR, YOLO) from falsely covering specific models (Issue 2)."""
    from backend.core.coverage import subject_matches_entity

    # Broad parent names must NOT match specific models
    assert not subject_matches_entity("DETR", "rt-detr")
    assert not subject_matches_entity("YOLO", "yolov8")
    assert not subject_matches_entity("v8", "yolov8")

    # Target models and variant suffixes MUST match
    assert subject_matches_entity("RT-DETR-R50", "rt-detr")
    assert subject_matches_entity("RT-DETR", "rt-detr")
    assert subject_matches_entity("YOLOv8", "yolov8")
    assert subject_matches_entity("YOLOv8n", "yolov8")
    assert subject_matches_entity("YOLOv8-L", "yolov8")


def test_ollama_backend_clamps_predict_to_stay_within_context():
    """Verifies OllamaBackend clamps num_predict so prompt + predict <= context_window (Issue 1 & 3)."""
    from backend.llm.ollama import OllamaBackend
    backend = OllamaBackend(context_window=2048, max_output_tokens=1024)

    # Prompt of ~1714 tokens (6000 chars)
    long_prompt = "test " * 1200
    num_ctx, num_predict = backend._resolve_context_and_predict(long_prompt, None, {})

    assert num_ctx == 2048
    # num_predict must be clamped from 1024 so prompt (~1714) + num_predict <= 2048
    assert num_predict < 1024
    assert num_predict + int(len(long_prompt) / 3.5) <= 2048


def test_ollama_backend_raises_when_prompt_exceeds_context():
    """Verifies OllamaBackend rejects prompts that exceed num_ctx or leave insufficient output room (Issue 3)."""
    from backend.llm.ollama import OllamaBackend
    from backend.core.errors import ModelInferenceError
    backend = OllamaBackend(context_window=2048, max_output_tokens=1024)

    # Prompt of ~2100 tokens (> 2048)
    huge_prompt = "test " * 1500
    with pytest.raises(ModelInferenceError) as exc_info:
        backend._resolve_context_and_predict(huge_prompt, None, {})
    assert "Prompt is too large" in str(exc_info.value)

    # Prompt leaving < 64 tokens headroom (e.g. 2000 tokens)
    tight_prompt = "test " * 1410
    with pytest.raises(ModelInferenceError) as exc_info:
        backend._resolve_context_and_predict(tight_prompt, None, {})
    assert "insufficient output headroom" in str(exc_info.value)


def test_arxiv_html_fallback_to_pdf_in_fetch(isolated_engine):
    """Verifies that 404/empty arXiv HTML automatically falls back to PDF (Issue 1 fix)."""
    engine = isolated_engine
    state = ResearchState(session_id="sess_fetch_fb", goal="Compare YOLO and DETR", phase=ResearchPhase.SEARCH)
    engine.session_repo.create(state.session_id, state.goal, phase="SEARCH", status="PENDING")

    url = "https://arxiv.org/html/2304.08069v1"

    # HTML fetch fails (e.g. 404)
    engine.fetch_tool.fetch = MagicMock(return_value=None)

    # PDF fetch succeeds
    from backend.tools.pdf_fetch import FetchedPDFContent
    pdf_mock = FetchedPDFContent(
        url="https://arxiv.org/pdf/2304.08069v1.pdf",
        title="DETRs Beat YOLOs",
        text="Real-time DETR achieves SOTA results on COCO benchmark.",
        content_hash="hash123",
        pages_count=10,
        file_path="data/pdf/dummy.pdf"
    )
    engine.pdf_tool.fetch = MagicMock(return_value=pdf_mock)

    docs = engine.run_fetch_phase(state, urls=[url])

    assert len(docs) == 1
    assert docs[0].text == pdf_mock.text
    engine.pdf_tool.fetch.assert_called_once_with("https://arxiv.org/pdf/2304.08069v1.pdf")

