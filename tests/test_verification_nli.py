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
    """Review 485345a item 2: If claims have citations but NONE are NLI SUPPORTED by LLM, session must be PARTIAL."""
    from backend.core.state import ResearchPhase

    db = DatabaseManager(db_path=tmp_path / "nli_done_test.db")
    mock_llm = MagicMock()
    mock_llm.model = "mock"
    mock_llm.generate.return_value = LLMResponse(
        content="YOLOv8 uses cross-stage partial connections with anchor-free detection heads [E1].",
        total_tokens=40
    )
    mock_llm.structured_generate.return_value = LLMResponse(
        content='{"label": "NOT_SUPPORTED", "confidence": 0.85, "reason": "Not supported by fox quote"}',
        parsed=NLIStructuredOutputSchema(
            label=NLILabel.NOT_SUPPORTED,
            confidence=0.85,
            reason="Not supported by fox quote"
        ),
        total_tokens=40
    )

    engine = ResearchEngine(
        llm=mock_llm,
        db=db,
        limits=ResearchLimits(max_research_steps=1),
        embedding_backend=LocalHashEmbeddingBackend(64)
    )

    state = engine.create_session("Evaluate YOLOv8")
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


def test_engine_done_records_nli_unavailable_when_llm_fails_or_missing(tmp_path: Path):
    """Review item 3: When LLM NLI is unavailable/fails, record 'NLI unavailable – claims not verified'."""
    from backend.core.state import ResearchPhase

    db = DatabaseManager(db_path=tmp_path / "nli_unavail_test.db")
    mock_llm = MagicMock()
    mock_llm.model = "mock"
    mock_llm.generate.return_value = LLMResponse(
        content="YOLOv8 achieves fast inference on standard benchmarks [E1].",
        total_tokens=40
    )
    # LLM structured_generate fails (simulating Ollama offline, timeout, or budget exceeded)
    mock_llm.structured_generate.side_effect = RuntimeError("Ollama connection refused")

    engine = ResearchEngine(
        llm=mock_llm,
        db=db,
        limits=ResearchLimits(max_research_steps=1),
        embedding_backend=LocalHashEmbeddingBackend(64)
    )

    state = engine.create_session("Evaluate YOLOv8")
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
        raw_quote="YOLOv8 attains high throughput on standard benchmarks."
    )
    engine.evidence_repo.add(
        evidence_id="evi_1",
        session_id=session_id,
        raw_evidence_id="raw_1",
        subject="yolov8",
        predicate="attains",
        object_data={"statement": "YOLOv8 attains high throughput."}
    )
    state.source_ids = ["src_1"]

    engine.run_basic_answer(state, fetched_docs=[], retrieved_chunks=[])

    assert state.phase == ResearchPhase.PARTIAL
    assert state.status.value == "PARTIAL"
    assert "NLI unavailable – claims not verified" in state.error_message


# --- 5. Review 69c1355 Tests: 5 Review Issues Fixes ---

def test_rule_based_nli_claim_containing_quote_plus_unverified_not_supported():
    """Issue 1 (P0): Claim containing quote + ungrounded claims must NOT be marked SUPPORTED 1.0."""
    verifier = RuleBasedNLIVerifier()
    claim = "PointPillars achieves 59.2 NDS and is 10x faster than CenterPoint and all other detectors"
    quote = "PointPillars achieves 59.2 NDS"

    result = verifier.verify(claim, quote)
    assert result.label != NLILabel.SUPPORTED
    assert result.label in (NLILabel.PARTIALLY_SUPPORTED, NLILabel.NOT_SUPPORTED)


def test_rule_based_nli_multi_quote_numeric_consistency_not_rejected():
    """Issue 2 (P0): Multi-quote claims must check numbers against the union of cited quotes."""
    verifier = RuleBasedNLIVerifier()
    claim = "CenterPoint reaches 67.3 NDS [E1] while PointPillars reaches 59.2 NDS [E2]"
    quotes = [
        "CenterPoint achieves 60.3 mAP and 67.3 NDS",
        "PointPillars achieves 59.2 NDS"
    ]

    # When all_quotes is passed, neither quote check should fail due to 'number not found'
    res1 = verifier.verify(claim, quotes[0], all_quotes=quotes)
    assert "not found" not in res1.reason.lower()

    res2 = verifier.verify(claim, quotes[1], all_quotes=quotes)
    assert "not found" not in res2.reason.lower()

    # In pipeline with combined quote evaluation:
    pipeline = ClaimVerificationPipeline(verifier=verifier)
    pipe_res = pipeline.verify_claim_against_quotes(claim, quotes)
    assert pipe_res.label != NLILabel.CONTRADICTED
    assert "not found" not in pipe_res.reason.lower()


def test_rule_based_nli_metric_polarity_not_falsely_contradicted():
    """Issue 3 (P1): 'lower error rate' vs 'higher accuracy' must NOT be falsely CONTRADICTED."""
    verifier = RuleBasedNLIVerifier()
    claim = "RT-DETR has a lower error rate than YOLOv8"
    quote = "RT-DETR achieves higher accuracy than YOLOv8"

    result = verifier.verify(claim, quote)
    assert result.label != NLILabel.CONTRADICTED

    # Contrasting test: same metric ('latency') with opposite direction MUST be CONTRADICTED
    claim_same_metric = "RT-DETR has lower latency than YOLOv8"
    quote_same_metric = "RT-DETR has higher latency than YOLOv8"
    result_same_metric = verifier.verify(claim_same_metric, quote_same_metric)
    assert result_same_metric.label == NLILabel.CONTRADICTED
    assert result_same_metric.confidence == 0.95


def test_extract_comparative_triples_ignores_numbers_and_units():
    """Issue 4 (P1): Quantifiers like '2x' and metric units like '1.5 AP' must not be grabbed as subject."""
    from backend.verification.nli import extract_comparative_triples

    text1 = "YOLOv8 is 2x faster than RT-DETR"
    triples1 = extract_comparative_triples(text1)
    assert len(triples1) == 1
    assert triples1[0][0] == "yolov8"
    assert triples1[0][1] == "faster"
    assert triples1[0][2] == "rt-detr"

    text2 = "On COCO val2017, RT-DETR is 1.5 AP higher than YOLOv8"
    triples2 = extract_comparative_triples(text2)
    assert len(triples2) == 1
    assert triples2[0][0] == "rt-detr"
    assert triples2[0][1] == "higher"
    assert triples2[0][2] == "yolov8"


def test_budget_tracker_record_llm_call_coercion_and_validation():
    """Issue 5 (P2): record_llm_call should coerce float tokens and raise ValueError on invalid types."""
    from backend.core.budget import ExecutionBudgetTracker
    from backend.core.limits import ResearchLimits

    tracker = ExecutionBudgetTracker(limits=ResearchLimits(max_llm_calls=10, max_tokens=1000))
    # Float tokens coerced safely
    tracker.record_llm_call(tokens=120.0, count=1)
    assert tracker.total_tokens_consumed == 120
    assert tracker.llm_calls == 1

    # String numeric coerced safely
    tracker.record_llm_call(tokens="80", count="2")
    assert tracker.total_tokens_consumed == 200
    assert tracker.llm_calls == 3

    # Invalid type raises ValueError
    with pytest.raises(ValueError):
        tracker.record_llm_call(tokens="invalid_token_count")


# --- 6. Review aa5a034 / Current Turn Tests: Polarity, Generic Nouns, & 1-Call Multi-Quote ---

def test_extract_comparative_triples_skips_generic_nouns_and_prioritizes_known_entities():
    """Item 4: 'RT-DETR is a real-time detector with higher AP than YOLOv8' must pick RT-DETR, not detector."""
    from backend.verification.nli import extract_comparative_triples

    text = "RT-DETR is a real-time detector with higher AP than YOLOv8"

    # Test without known entities: generic nouns like 'detector' and 'real-time' must be skipped
    triples_no_hint = extract_comparative_triples(text)
    assert len(triples_no_hint) == 1
    assert triples_no_hint[0][0] == "rt-detr"
    assert triples_no_hint[0][1] == "higher"
    assert triples_no_hint[0][2] == "yolov8"

    # Test with explicit known research entities: prioritizes goal entities
    triples_with_hint = extract_comparative_triples(text, known_entities=["rt-detr", "yolov8"])
    assert len(triples_with_hint) == 1
    assert triples_with_hint[0][0] == "rt-detr"
    assert triples_with_hint[0][1] == "higher"
    assert triples_with_hint[0][2] == "yolov8"


def test_metric_polarity_inherent_and_derived():
    """Item 4: _get_metric_polarity and RuleBasedNLIVerifier handle lower-is-better and higher-is-better metrics."""
    from backend.verification.nli import _get_metric_polarity

    # Polarity helper verification
    assert _get_metric_polarity("faster", "") == 1
    assert _get_metric_polarity("slower", "") == -1
    assert _get_metric_polarity("higher", "accuracy") == 1
    assert _get_metric_polarity("lower", "accuracy") == -1
    assert _get_metric_polarity("higher", "latency") == -1
    assert _get_metric_polarity("lower", "latency") == 1
    assert _get_metric_polarity("higher", "error rate") == -1
    assert _get_metric_polarity("lower", "error rate") == 1

    verifier = RuleBasedNLIVerifier()

    # Opposite polarity in speed domain: higher latency (-1) vs faster (+1) -> CONTRADICTED
    c1 = "RT-DETR has higher latency than YOLOv8"
    q1 = "RT-DETR is faster than YOLOv8"
    res1 = verifier.verify(c1, q1)
    assert res1.label == NLILabel.CONTRADICTED
    assert res1.confidence == 0.95

    # Opposite polarity in accuracy domain: higher error rate (-1) vs higher accuracy (+1) -> CONTRADICTED
    c2 = "RT-DETR has higher error rate than YOLOv8"
    q2 = "RT-DETR achieves higher accuracy than YOLOv8"
    res2 = verifier.verify(c2, q2)
    assert res2.label == NLILabel.CONTRADICTED
    assert res2.confidence == 0.95

    # Aligned polarity: lower error rate (+1) vs higher accuracy (+1) -> NOT CONTRADICTED
    c3 = "RT-DETR has lower error rate than YOLOv8"
    q3 = "RT-DETR achieves higher accuracy than YOLOv8"
    res3 = verifier.verify(c3, q3)
    assert res3.label != NLILabel.CONTRADICTED


def test_pipeline_multi_quote_single_llm_call_and_no_single_quote_contradiction():
    """Item 2: Multi-quote claims run rule checks on individual quotes (0 LLM calls) and call LLM exactly once on combined quotes."""
    mock_llm = MagicMock(spec=LLMBackend)
    mock_schema_res = NLIStructuredOutputSchema(
        label=NLILabel.SUPPORTED,
        confidence=0.95,
        reason="Evidence confirms both entities' results."
    )
    mock_llm.structured_generate.return_value = LLMResponse(
        content='{"label": "SUPPORTED", "confidence": 0.95, "reason": "Evidence confirms both entities\' results."}',
        parsed=mock_schema_res,
        total_tokens=100,
        calls_made=1
    )

    composite = CompositeNLIVerifier(llm=mock_llm)
    pipeline = ClaimVerificationPipeline(verifier=composite)

    quotes = [
        "CenterPoint achieves 67.3 NDS on nuScenes.",
        "PointPillars achieves 59.2 NDS on nuScenes."
    ]
    claim = "CenterPoint reaches 67.3 NDS while PointPillars reaches 59.2 NDS on nuScenes."

    res = pipeline.verify_claim_against_quotes(claim, quotes)

    assert res.label == NLILabel.SUPPORTED
    assert res.confidence == 0.95
    # Must make EXACTLY ONE call to LLM structured_generate on combined evidence, NOT 3 calls (N+1)!
    assert mock_llm.structured_generate.call_count == 1

    call_args, call_kwargs = mock_llm.structured_generate.call_args
    prompt_used = call_args[0]
    # Prompt must contain both quotes combined so the 3B LLM has full context
    assert "CenterPoint achieves 67.3 NDS" in prompt_used
    assert "PointPillars achieves 59.2 NDS" in prompt_used

