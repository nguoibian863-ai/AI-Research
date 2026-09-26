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
from backend.evidence.extractor import EvidenceExtractor
from backend.llm.schemas import ExtractedEvidencesSchema, AtomicFactItemSchema
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
    assert item.url == "https://arxiv.org/abs/2304.08069v1"
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


def test_openalex_search_provider_offline():
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
    assert item.url == "https://dl.acm.org/doi/10.1145/123456.7890"
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


def test_composite_search_provider_deduplication():
    class FakeProviderA:
        name = "provider_a"
        def search(self, query, max_results=8):
            return [
                SearchResultItem(
                    title="YOLOv8 Paper",
                    url="https://arxiv.org/abs/2305.09972",
                    snippet="Official YOLOv8 paper",
                    source_type="paper"
                )
            ]

    class FakeProviderB:
        name = "provider_b"
        def search(self, query, max_results=8):
            return [
                # Same arXiv paper in pdf format -> deduplicated by canonical key
                SearchResultItem(
                    title="YOLOv8 Paper PDF",
                    url="https://arxiv.org/pdf/2305.09972.pdf",
                    snippet="PDF version of YOLOv8 paper",
                    source_type="paper"
                ),
                SearchResultItem(
                    title="Ultralytics YOLO Documentation",
                    url="https://docs.ultralytics.com/models/yolov8",
                    snippet="Documentation page",
                    source_type="web"
                )
            ]

    composite = CompositeSearchProvider(providers=[FakeProviderA(), FakeProviderB()], max_results=5)
    results = composite.search("YOLOv8")

    assert len(results) == 2
    assert results[0].url == "https://arxiv.org/abs/2305.09972"
    assert results[1].url == "https://docs.ultralytics.com/models/yolov8"


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

    # 1. Run Search Phase -> generates query_generation trajectory
    engine.run_search_phase(state, custom_queries=["YOLOv8 RT-DETR COCO benchmark"])

    query_trajs = engine.trajectory_repo.list_by_task_type("query_generation", partition="raw")
    assert len(query_trajs) >= 1
    qt = query_trajs[0]
    payload = json.loads(qt["payload_json"])
    assert payload["goal"] == state.goal
    assert "queries" in payload
    assert "relevant_sources_count" in payload
    assert qt["verified"] in (0, 1)

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

    # Verify trajectory table recorded BOTH facts (VERIFIED and QUOTE_NOT_FOUND)
    ext_trajs = engine.trajectory_repo.list_by_task_type("evidence_extraction", partition="raw")
    assert len(ext_trajs) >= 2

    verdicts = [json.loads(t["payload_json"])["verdict"] for t in ext_trajs]
    assert "VERIFIED" in verdicts
    assert "QUOTE_NOT_FOUND" in verdicts

    # 3. Test JSONL export per Plan 43.1 / 42.4
    counts = engine.trajectory_repo.export_partition_jsonl(partition="raw", output_dir=tmp_path / "trajectories")
    assert counts.get("query_generation", 0) >= 1
    assert counts.get("evidence_extraction", 0) >= 2

    jsonl_file = tmp_path / "trajectories" / "raw" / "evidence_extraction.jsonl"
    assert jsonl_file.exists()
    lines = [json.loads(line) for line in jsonl_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 2
    for line in lines:
        assert "trajectory_id" in line
        assert "metadata" in line
        assert "model_source" in line["metadata"]
