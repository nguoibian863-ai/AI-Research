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


# --- 1. Review Item 2 Tests: Accurate Rule-Based NLI on Comparative & Negation Cases ---

def test_rule_based_nli_identical_antonyms_with_citation_supported():
    """Review 485345a case 1: Claim identical to quote even with [E1] citation tag must be SUPPORTED 1.0."""
    verifier = RuleBasedNLIVerifier()
    claim = "RT-DETR achieves higher AP with lower latency than YOLOv8 [E1]"
    quote = "RT-DETR achieves higher AP with lower latency than YOLOv8"

    result = verifier.verify(claim, quote)
    assert result.label == NLILabel.SUPPORTED
    assert result.confidence == 1.0


def test_rule_based_nli_without_not_negation_contradiction():
    """Preposition 'without' must NOT be flagged as a verbal negation contradiction."""
    verifier = RuleBasedNLIVerifier()
    claim = "RT-DETR removes NMS"
    quote = "RT-DETR operates without NMS"

    result = verifier.verify(claim, quote)
    assert result.label in (NLILabel.SUPPORTED, NLILabel.PARTIALLY_SUPPORTED)
    assert result.label != NLILabel.CONTRADICTED


def test_rule_based_nli_reversed_comparison_is_contradicted():
    """Swapped entities around comparative direction MUST be caught as CONTRADICTED."""
    verifier = RuleBasedNLIVerifier()
    claim = "YOLOv8 is faster than RT-DETR"
    quote = "RT-DETR is faster than YOLOv8"

    result = verifier.verify(claim, quote)
    assert result.label == NLILabel.CONTRADICTED
    assert result.confidence == 0.95


def test_rule_based_nli_introductory_context_reversed_is_contradicted():
    """Review 485345a case 2: 'On T4 GPU, YOLOv8 is faster than RT-DETR' must detect YOLOv8 as subject."""
    verifier = RuleBasedNLIVerifier()
    claim = "On T4 GPU, YOLOv8 is faster than RT-DETR"
    quote = "RT-DETR is faster than YOLOv8"

    result = verifier.verify(claim, quote)
    assert result.label == NLILabel.CONTRADICTED
    assert result.confidence == 0.95


def test_rule_based_nli_verbal_negation_defers_to_llm():
    """Review 485345a case 3: 'RT-DETR is not faster than YOLOv8' has negation mismatch, must not be SUPPORTED."""
    verifier = RuleBasedNLIVerifier()
    claim = "RT-DETR is not faster than YOLOv8"
    quote = "RT-DETR is faster than YOLOv8"

    result = verifier.verify(claim, quote)
    assert result.label == NLILabel.NOT_SUPPORTED
    assert result.confidence <= 0.70
    assert "verbal negation" in result.reason.lower()


def test_rule_based_nli_model_variants_reversed_is_contradicted():
    """Review 485345a case 4: 'YOLOv8 runs faster than RT-DETR-R50' vs 'RT-DETR-R50 runs faster than YOLOv8-L'."""
    verifier = RuleBasedNLIVerifier()
    claim = "YOLOv8 runs faster than RT-DETR-R50"
    quote = "RT-DETR-R50 runs faster than YOLOv8-L"

    result = verifier.verify(claim, quote)
    assert result.label == NLILabel.CONTRADICTED
    assert result.confidence == 0.95


def test_extract_comparative_triples_word_boundaries():
    """Review 485345a item 1e: Substrings 'more' in 'furthermore' and 'less' in 'lossless' must not match."""
    from backend.verification.nli import extract_comparative_triples
    text = "Furthermore, lossless compression is preserved."
    triples = extract_comparative_triples(text)
    assert len(triples) == 0


def test_rule_based_nli_opposing_comparative_is_contradicted():
    """Opposing comparative adjectives on the same entity pair MUST be caught as CONTRADICTED."""
    verifier = RuleBasedNLIVerifier()
    claim = "PointPillars is faster than CenterPoint"
    quote = "PointPillars is slower than CenterPoint"

    result = verifier.verify(claim, quote)
    assert result.label == NLILabel.CONTRADICTED
    assert "opposite" in result.reason.lower() or "contradiction" in result.reason.lower()


def test_rule_based_nli_token_overlap_capped_does_not_bypass_llm():
    """Token overlap without exact match returns PARTIALLY_SUPPORTED (conf <= 0.70) to never bypass LLM."""
    verifier = RuleBasedNLIVerifier()
    claim = "CenterPoint achieves 67.3 NDS on the nuScenes detection benchmark."
    evidence = "Our CenterPoint model achieves 67.3 NDS on nuScenes 3D detection benchmark."

    result = verifier.verify(claim, evidence)
    assert result.label == NLILabel.PARTIALLY_SUPPORTED
    assert result.confidence <= 0.75  # < 0.85 threshold, ensuring CompositeNLIVerifier calls LLM!
    assert result.verifier_type == "rule_based"


def test_rule_based_nli_not_supported_low_overlap():
    verifier = RuleBasedNLIVerifier()
    claim = "YOLOv8 uses cross-stage partial connections with anchor-free detection heads."
    evidence = "PostgreSQL stores tables in 8KB blocks on NVMe disk arrays."

    result = verifier.verify(claim, evidence)
    assert result.label == NLILabel.NOT_SUPPORTED
    assert "insufficient" in result.reason.lower()


# --- 2. Review Item 4 Tests: Budget Enforcement for LLM NLI Calls ---

def test_llm_nli_verifier_records_budget():
    mock_llm = MagicMock(spec=LLMBackend)
    mock_schema_res = NLIStructuredOutputSchema(
        label=NLILabel.SUPPORTED,
        confidence=0.95,
        reason="Evidence confirms the metric and dataset exactly."
    )
    mock_llm.structured_generate.return_value = LLMResponse(
        content='{"label": "SUPPORTED", "confidence": 0.95, "reason": "Evidence confirms the metric and dataset exactly."}',
        parsed=mock_schema_res,
        total_tokens=85,
        calls_made=1
    )

    mock_budget = MagicMock()
    mock_budget.assert_can_call_llm = MagicMock()
    mock_budget.record_llm_call = MagicMock()
    verifier = LLMNLIVerifier(llm=mock_llm, budget_tracker=mock_budget)
    claim = "YOLOv8-X reaches 53.9 mAP on COCO."
    evidence = "The largest variant YOLOv8-X attains 53.9 mAP on COCO test-dev."

    res = verifier.verify(claim, evidence)
    assert res.label == NLILabel.SUPPORTED
    assert res.confidence == 0.95
    assert res.verifier_type == "llm"

    # Verifies that LLM call was recorded in budget tracker (Review Item 4)
    assert mock_budget.assert_can_call_llm.call_count == 1
    assert mock_budget.record_llm_call.call_count == 1
    mock_budget.record_llm_call.assert_called_with(tokens=85, count=1)


def test_llm_nli_verifier_fallback_on_budget_exhaustion():
    mock_llm = MagicMock(spec=LLMBackend)
    mock_budget = MagicMock()
    mock_budget.assert_can_call_llm = MagicMock(side_effect=RuntimeError("Budget exhausted"))

    verifier = LLMNLIVerifier(llm=mock_llm, budget_tracker=mock_budget)
    claim = "CenterPoint achieves 67.3 NDS on nuScenes."
    evidence = "CenterPoint achieves 67.3 NDS on nuScenes benchmark."

    res = verifier.verify(claim, evidence)
    # When budget check fails, falls back immediately to zero-cost rule-based verifier!
    assert res.label == NLILabel.SUPPORTED
    assert res.verifier_type == "rule_based"
    assert mock_llm.structured_generate.call_count == 0


def test_claim_verification_pipeline_multi_quotes():
    pipeline = ClaimVerificationPipeline()
    claim = "CenterPoint achieves 67.3 NDS on nuScenes."

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


# --- 3. Review Item 1 & 3 Tests: Status Preservation & Contradiction Session Transition ---

def test_verify_session_claims_preserves_numeric_mismatch_and_status(tmp_path: Path):
    """Review item 3: verify_session_claims MUST NOT overwrite NUMERIC_MISMATCH with SUPPORTED."""
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

    # Claim 1: CITED
    engine.claim_repo.add(
        claim_id="clm_1",
        session_id=session_id,
        text="CenterPoint achieves 67.3 NDS on the nuScenes detection benchmark [E1].",
        evidence_ids=["evi_cp"],
        verification={
            "verified": False,
            "entailment": "PENDING",
            "numeric_match": True,
            "citation_indices": [1]
        },
        status="CITED"
    )

    # Claim 2: NUMERIC_MISMATCH
    engine.claim_repo.add(
        claim_id="clm_2",
        session_id=session_id,
        text="CenterPoint achieves 99.9 NDS on nuScenes detection benchmark [E1].",
        evidence_ids=["evi_cp"],
        verification={
            "verified": False,
            "numeric_match": False,
            "citation_indices": [1],
            "mismatched_numbers": ["99.9"]
        },
        status="NUMERIC_MISMATCH"
    )

    # Run verification
    updated = engine.verify_session_claims(session_id)
    assert len(updated) == 1  # Only CITED claim was verified, NUMERIC_MISMATCH was skipped!

    # Claim 1 verification updated, status remains CITED
    c1 = engine.claim_repo.get_by_id("clm_1")
    assert c1["status"] == "CITED"
    assert c1["verification"]["verified"] is True
    assert c1["verification"]["entailment"] == "SUPPORTED"

    # Claim 2 MUST remain NUMERIC_MISMATCH
    c2 = engine.claim_repo.get_by_id("clm_2")
    assert c2["status"] == "NUMERIC_MISMATCH"
    assert c2["verification"]["verified"] is False
    assert c2["verification"]["numeric_match"] is False


def test_contradiction_in_report_triggers_session_partial(tmp_path: Path):
    """Review item 3: If report contains a claim CONTRADICTED by evidence, session must transition to PARTIAL."""
    from backend.core.state import ResearchPhase

    db = DatabaseManager(db_path=tmp_path / "contra_test.db")
    mock_llm = MagicMock()
    mock_llm.model = "mock"
    # Report contains a comparative claim that directly contradicts the evidence quote!
    mock_llm.generate.return_value = LLMResponse(
        content="YOLOv8 is faster than RT-DETR [E1].",
        total_tokens=40
    )

    engine = ResearchEngine(
        llm=mock_llm,
        db=db,
        limits=ResearchLimits(max_research_steps=1),
        embedding_backend=LocalHashEmbeddingBackend(64)
    )

    state = engine.create_session("Compare YOLOv8 and RT-DETR")
    state.phase = ResearchPhase.EVALUATE
    session_id = state.session_id

    # Evidence quote states the reverse: RT-DETR is faster than YOLOv8!
    engine.source_repo.add(
        source_id="src_1",
        session_id=session_id,
        url="https://arxiv.org/html/2304.08069",
        title="RT-DETR paper",
        domain="arxiv.org"
    )
    engine.raw_evidence_repo.add(
        raw_evidence_id="raw_contra",
        session_id=session_id,
        source_id="src_1",
        raw_quote="RT-DETR is faster than YOLOv8 on COCO benchmark."
    )
    engine.evidence_repo.add(
        evidence_id="evi_contra",
        session_id=session_id,
        raw_evidence_id="raw_contra",
        subject="rt-detr",
        predicate="is faster than",
        object_data={"statement": "RT-DETR is faster than YOLOv8 on COCO."}
    )
    state.source_ids = ["src_1"]

    answer = engine.run_basic_answer(state, fetched_docs=[], retrieved_chunks=[])

    assert state.phase == ResearchPhase.PARTIAL
    assert state.status.value == "PARTIAL"
    assert "Report contains claims contradicted by cited evidence" in state.error_message

    claims = engine.claim_repo.get_by_session(session_id)
    assert len(claims) == 1
    assert claims[0]["verification"]["entailment"] == "CONTRADICTED"
    assert claims[0]["verification"]["verified"] is False


def test_engine_done_requires_at_least_one_nli_supported_claim(tmp_path: Path):
    """Review 485345a item 2: If claims have citations but NONE are NLI SUPPORTED, session must be PARTIAL."""
    from backend.core.state import ResearchPhase

    db = DatabaseManager(db_path=tmp_path / "nli_done_test.db")
    mock_llm = MagicMock()
    mock_llm.model = "mock"
    # Claim text has low overlap with quote and is NOT exact match -> NLI NOT_SUPPORTED
    mock_llm.generate.return_value = LLMResponse(
        content="YOLOv8 uses cross-stage partial connections with anchor-free detection heads [E1].",
        total_tokens=40
    )

    engine = ResearchEngine(
        llm=mock_llm,
        db=db,
        limits=ResearchLimits(max_research_steps=1),
        embedding_backend=LocalHashEmbeddingBackend(64)
    )

    state = engine.create_session("Evaluate YOLOv8 architecture")
    state.phase = ResearchPhase.EVALUATE
    session_id = state.session_id

    engine.source_repo.add(
        source_id="src_1",
        session_id=session_id,
        url="https://github.com/ultralytics/ultralytics",
        title="YOLOv8 GitHub",
        domain="github.com"
    )
    engine.raw_evidence_repo.add(
        raw_evidence_id="raw_1",
        session_id=session_id,
        source_id="src_1",
        raw_quote="The quick brown fox jumps over the lazy dog."
    )
    engine.evidence_repo.add(
        evidence_id="evi_1",
        session_id=session_id,
        raw_evidence_id="raw_1",
        subject="yolov8",
        predicate="uses",
        object_data={"statement": "YOLOv8 architecture is modern."}
    )
    state.source_ids = ["src_1"]

    answer = engine.run_basic_answer(state, fetched_docs=[], retrieved_chunks=[])

    # MUST be PARTIAL, not DONE!
    assert state.phase == ResearchPhase.PARTIAL
    assert state.status.value == "PARTIAL"
    assert "none were confirmed as fully supported by evidence" in state.error_message

    claims = engine.claim_repo.get_by_session(session_id)
    assert len(claims) == 1
    assert claims[0]["status"] == "CITED"
    assert claims[0]["verification"]["verified"] is False
    assert claims[0]["verification"]["entailment"] == "NOT_SUPPORTED"
