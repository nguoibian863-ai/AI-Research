"""
NLI Calibration Runner: Evaluates CompositeNLIVerifier against 20 benchmark pairs using local Ollama model.
"""
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.llm.ollama import OllamaBackend
from backend.verification.nli import CompositeNLIVerifier

def run_calibration(fixture_path: Path = Path("tests/fixtures/nli_calibration.jsonl"), model_name: str = "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M"):
    if not fixture_path.exists():
        print(f"Error: Fixture {fixture_path} not found.")
        return None

    pairs = [json.loads(line) for line in fixture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    llm = OllamaBackend(model=model_name, system_prefix="/no_think", timeout=120.0)
    composite = CompositeNLIVerifier(llm=llm)

    print(f"Total calibration pairs: {len(pairs)}", flush=True)
    results = []
    false_positives = 0
    negative_count = 0

    for p in pairs:
        res = composite.verify(p["claim"], p["evidence"])
        got_label = res.label.value
        expected = p["label"]
        is_neg = (expected in ("NOT_SUPPORTED", "CONTRADICTED"))
        is_fp = is_neg and (got_label == "SUPPORTED")
        if is_neg:
            negative_count += 1
            if is_fp:
                false_positives += 1

        matched = (got_label == expected)
        results.append({
            "id": p["id"],
            "expected": expected,
            "got": got_label,
            "confidence": res.confidence,
            "verifier_type": res.verifier_type,
            "matched": matched,
            "reason": res.reason
        })
        p_id = p["id"]
        v_type = res.verifier_type
        print(f"#{p_id:02d} | Exp: {expected:<15} | Got: {got_label:<15} ({res.confidence:.2f} by {v_type:<10}) | Match: {matched}", flush=True)

    fp_rate = (false_positives / negative_count) * 100 if negative_count else 0
    print("\n--- CALIBRATION SUMMARY ---", flush=True)
    print(f"False Positive SUPPORTED rate on negatives: {false_positives}/{negative_count} ({fp_rate:.1f}%)", flush=True)

    by_label = {}
    for r in results:
        exp = r["expected"]
        by_label.setdefault(exp, {"total": 0, "correct": 0})
        by_label[exp]["total"] += 1
        if r["matched"]:
            by_label[exp]["correct"] += 1

    for lbl, st in by_label.items():
        acc = (st["correct"] / st["total"]) * 100
        corr = st["correct"]
        tot = st["total"]
        print(f"Label {lbl:<15}: {corr}/{tot} correct ({acc:.1f}%)", flush=True)

    overall_acc = (sum(st["correct"] for st in by_label.values()) / len(results)) * 100
    print(f"Overall Accuracy: {overall_acc:.1f}%\n", flush=True)
    return {
        "results": results,
        "false_positive_rate": fp_rate,
        "by_label": by_label,
        "overall_accuracy": overall_acc
    }

if __name__ == "__main__":
    run_calibration()
