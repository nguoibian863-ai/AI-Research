import pytest
from unittest.mock import MagicMock
from pathlib import Path

from backend.verification.schemas import NLILabel, NLIVerificationResult, NLIStructuredOutputSchema
from backend.verification.nli import (
    RuleBasedNLIVerifier,
    LLMNLIVerifier,
    CompositeNLIVerifier,
    ClaimVerificationPipeline
)
from backend.db.database import DatabaseManager
from backend.llm.backend import LLMResponse, LLMBackend
from backend.core.engine import ResearchEngine
from backend.core.limits import ResearchLimits
from backend.retrieval.embeddings import LocalHashEmbeddingBackend


def test_rule_based_nli_supported_high_overlap():
    verifier = RuleBasedNLIVerifier()
    claim = "CenterPoint achieves 67.3 NDS on the nuScenes detection benchmark."
    evidence = "Our CenterPoint model achieves 67.3 NDS on nuScenes 3D detection benchmark."

    result = verifier.verify(claim, evidence)
    assert result.label == NLILabel.SUPPORTED
    assert result.confidence >= 0.8
    assert "token alignment" in result.reason.lower()
    assert result.verifier_type == "rule_based"


def test_rule_based_nli_directional_contradiction():
    verifier = RuleBasedNLIVerifier()
    # Claim states 'faster' but evidence states 'slower'
    claim = "PointPillars is faster than CenterPoint on lidar perception."
    evidence = "In our experiments, PointPillars is slower than CenterPoint on lidar perception."

    result = verifier.verify(claim, evidence)
    assert result.label == NLILabel.CONTRADICTED
    assert "contradiction" in result.reason.lower()
    assert "faster" in result.reason
    assert "slower" in result.reason


def test_rule_based_nli_negation_contradiction():
    verifier = RuleBasedNLIVerifier()
    claim = "PostgreSQL cannot handle high concurrency workloads on TPC-C."
    evidence = "PostgreSQL can handle high concurrency workloads on TPC-C with multi-version concurrency control."

    result = verifier.verify(claim, evidence)
    assert result.label == NLILabel.CONTRADICTED
    assert "negation" in result.reason.lower()


def test_rule_based_nli_not_supported_low_overlap():
    verifier = RuleBasedNLIVerifier()
    claim = "YOLOv8 uses cross-stage partial connections with anchor-free detection heads."
    evidence = "PostgreSQL stores tables in 8KB blocks on NVMe disk arrays."

    result = verifier.verify(claim, evidence)
    assert result.label == NLILabel.NOT_SUPPORTED
    assert "insufficient" in result.reason.lower()


def test_rule_based_nli_partially_supported():
    verifier = RuleBasedNLIVerifier()
    claim = "RT-DETR achieves 53.1 AP on COCO with 108 FPS on T4 GPU."
    evidence = "RT-DETR reaches 53.1 AP on the COCO validation dataset."

    result = verifier.verify(claim, evidence)
    assert result.label in (NLILabel.PARTIALLY_SUPPORTED, NLILabel.SUPPORTED)


def test_llm_nli_verifier_structured():
    mock_llm = MagicMock(spec=LLMBackend)
    mock_schema_res = NLIStructuredOutputSchema(
        label=NLILabel.SUPPORTED,
        confidence=0.95,
        reason="Evidence confirms the metric and dataset exactly."
    )
    mock_llm.structured_generate.return_value = LLMResponse(
        content='{"label": "SUPPORTED", "confidence": 0.95, "reason": "Evidence confirms the metric and dataset exactly."}',
        parsed=mock_schema_res
    )

    verifier = LLMNLIVerifier(llm=mock_llm)
    claim = "YOLOv8-X reaches 53.9 mAP on COCO."
    evidence = "The largest variant YOLOv8-X attains 53.9 mAP on COCO test-dev."

    res = verifier.verify(claim, evidence)
    assert res.label == NLILabel.SUPPORTED
    assert res.confidence == 0.95
    assert res.verifier_type == "llm"


def test_llm_nli_verifier_fallback_on_error():
    mock_llm = MagicMock(spec=LLMBackend)
    mock_llm.structured_generate.side_effect = RuntimeError("Ollama connection timeout")

    verifier = LLMNLIVerifier(llm=mock_llm)
    claim = "CenterPoint achieves 67.3 NDS on nuScenes."
    evidence = "CenterPoint achieves 67.3 NDS on nuScenes benchmark."

    res = verifier.verify(claim, evidence)
    # Gracefully falls back to rule-based verifier!
    assert res.label == NLILabel.SUPPORTED
    assert res.verifier_type == "rule_based"


def test_claim_verification_pipeline_multi_quotes():
    pipeline = ClaimVerificationPipeline()
    claim = "CenterPoint achieves 67.3 NDS on nuScenes."

    # Multiple quotes where one supports and one is unrelated
    quotes = [
        "Unrelated text about autonomous driving cameras.",
        "CenterPoint achieves 67.3 NDS on nuScenes benchmark."
    ]
    res = pipeline.verify_claim_against_quotes(claim, quotes)
    assert res.label == NLILabel.SUPPORTED

    # If any quote directly contradicts -> CONTRADICTED takes precedence
    contradicting_quotes = [
        "CenterPoint achieves 67.3 NDS on nuScenes benchmark.",
        "CenterPoint trails PointPillars, achieving a lower score on nuScenes."
    ]
    contra_claim = "CenterPoint exceeds PointPillars on nuScenes."
    res_contra = pipeline.verify_claim_against_quotes(contra_claim, contradicting_quotes)
    assert res_contra.label == NLILabel.CONTRADICTED


def test_engine_verify_session_claims_updates_db(tmp_path: Path):
    db = DatabaseManager(db_path=tmp_path / "nli_test.db")
    engine = ResearchEngine(
        llm=MagicMock(),
        db=db,
        limits=ResearchLimits(),
        embedding_backend=LocalHashEmbeddingBackend(64)
    )

    state = engine.create_session("Compare CenterPoint and PointPillars on nuScenes")
    session_id = state.session_id

    # Add source & evidence
    engine.source_repo.add(
        source_id="src_cp",
        session_id=session_id,
        url="https://arxiv.org/html/2006.11275",
        title="CenterPoint paper",
        domain="arxiv.org"
    )
    engine.raw_evidence_repo.add(
        raw_evidence_id="raw_cp",
        session_id=session_id,
        source_id="src_cp",
        raw_quote="CenterPoint achieves 67.3 NDS on the nuScenes detection benchmark."
    )
    engine.evidence_repo.add(
        evidence_id="evi_cp",
        session_id=session_id,
        raw_evidence_id="raw_cp",
        subject="centerpoint",
        predicate="achieves",
        object_data={
            "metric": "NDS",
            "value": 67.3,
            "unit": "NDS",
            "statement": "CenterPoint achieves 67.3 NDS on nuScenes."
        },
        confidence=0.95
    )

    # Add claim in CITED status (as left after run_basic_answer)
    engine.claim_repo.add(
        claim_id="clm_1",
        session_id=session_id,
        text="CenterPoint achieves 67.3 NDS on nuScenes detection benchmark [E1].",
        evidence_ids=["evi_cp"],
        verification={
            "verified": False,
            "entailment": "PENDING",
            "numeric_match": True,
            "citation_indices": [1]
        },
        status="CITED"
    )

    # Run Week 4 session claims verification
    updated = engine.verify_session_claims(session_id)
    assert len(updated) == 1
    assert updated[0]["status"] == "SUPPORTED"
    assert updated[0]["verification"]["verified"] is True
    assert updated[0]["verification"]["entailment"] == "SUPPORTED"
    assert updated[0]["verification"]["entailment_confidence"] >= 0.8
    assert "token alignment" in updated[0]["verification"]["entailment_reason"].lower()
