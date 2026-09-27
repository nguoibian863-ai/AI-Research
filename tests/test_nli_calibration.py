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
    assert len(pairs) == 30, f"Expected 30 calibration pairs, got {len(pairs)}"

    labels = [p["label"] for p in pairs]
    assert labels.count("SUPPORTED") == 10
    assert labels.count("NOT_SUPPORTED") == 11
    assert labels.count("CONTRADICTED") == 9

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

    llm_m = summary["llm_metrics"]
    assert llm_m["false_positive_rate"] < 20.0, f"LLM standalone FP rate {llm_m['false_positive_rate']}% exceeds 20% threshold"
    assert llm_m["contra_caught"] >= 4, f"LLM standalone caught only {llm_m['contra_caught']} CONTRADICTED pairs"

    comp_m = summary["composite_metrics"]
    assert comp_m["false_positive_rate"] < 20.0, f"Composite FP rate {comp_m['false_positive_rate']}% exceeds 20%"
    assert comp_m["overall_accuracy"] >= 75.0, f"Composite accuracy {comp_m['overall_accuracy']}% below 75%"
