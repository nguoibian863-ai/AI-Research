import json
import pytest
from pathlib import Path
from backend.verification.nli import CompositeNLIVerifier
from backend.verification.schemas import NLILabel
from scripts.calibrate_nli import run_calibration


def test_nli_calibration_fixture_structure():
    fixture_path = Path("tests/fixtures/nli_calibration.jsonl")
    assert fixture_path.exists(), "Calibration fixture missing"
    pairs = [json.loads(line) for line in fixture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(pairs) == 20, f"Expected 20 calibration pairs, got {len(pairs)}"

    labels = [p["label"] for p in pairs]
    assert labels.count("SUPPORTED") == 8
    assert labels.count("NOT_SUPPORTED") == 6
    assert labels.count("CONTRADICTED") == 6

    for p in pairs:
        assert "claim" in p and len(p["claim"]) > 5
        assert "evidence" in p and len(p["evidence"]) > 5
        assert p["label"] in ("SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED")


def test_nli_calibration_rule_subset_precision():
    """
    Verifies that the rule-based component never produces false positive SUPPORTED on negative benchmark pairs.
    """
    fixture_path = Path("tests/fixtures/nli_calibration.jsonl")
    pairs = [json.loads(line) for line in fixture_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    composite = CompositeNLIVerifier(llm=None)
    neg_pairs = [p for p in pairs if p["label"] in ("NOT_SUPPORTED", "CONTRADICTED")]

    for p in neg_pairs:
        res = composite.verify(p["claim"], p["evidence"])
        # Rule verifier must never produce SUPPORTED for any negative benchmark pair
        assert res.label != NLILabel.SUPPORTED, f"False positive SUPPORTED on pair #{p['id']}: {p['claim']}"


@pytest.mark.slow
def test_nli_smollm3_calibration_benchmark():
    """
    Runs full calibration benchmark using local Ollama model if available.
    Asserts False Positive rate on negative pairs is strictly under 20%.
    """
    import httpx
    try:
        r = httpx.get("http://127.0.0.1:11434/", timeout=2.0)
        if r.status_code != 200:
            pytest.skip("Ollama server not running on localhost:11434")
    except Exception:
        pytest.skip("Ollama server unreachable")

    summary = run_calibration()
    assert summary is not None
    assert summary["false_positive_rate"] < 20.0, f"FP rate {summary['false_positive_rate']}% exceeds 20% threshold"
    assert summary["overall_accuracy"] >= 80.0, f"Overall accuracy {summary['overall_accuracy']}% below 80%"
