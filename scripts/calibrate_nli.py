"""
NLI Calibration Runner: Evaluates standalone LLMNLIVerifier and CompositeNLIVerifier
against 30 benchmark pairs using local Ollama model.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.llm.ollama import OllamaBackend
from backend.verification.nli import LLMNLIVerifier, RuleBasedNLIVerifier
from backend.verification.schemas import NLILabel


def run_calibration(
    fixture_path: Path = Path("tests/fixtures/nli_calibration.jsonl"),
    model_name: str = "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M"
) -> Optional[Dict[str, Any]]:
    if not fixture_path.exists():
        print(f"Error: Fixture {fixture_path} not found.", flush=True)
        return None

    pairs = [json.loads(line) for line in fixture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(pairs)} calibration pairs from {fixture_path}\n", flush=True)

    llm = OllamaBackend(model=model_name, system_prefix="/no_think", timeout=120.0)
    rule_verifier = RuleBasedNLIVerifier()
    llm_verifier = LLMNLIVerifier(llm=llm, fallback=rule_verifier)

    llm_results = []
    composite_results = []

    print("==================================================================", flush=True)
    print(" 1. STANDALONE LLM EVALUATION (Direct LLMNLIVerifier, bypassing rules)", flush=True)
    print("==================================================================", flush=True)

    for p in pairs:
        expected = p["label"]
        p_id = p["id"]

        # 1. Run LLM verifier directly
        llm_res = llm_verifier.verify(p["claim"], p["evidence"])
        llm_got = llm_res.label.value
        llm_matched = (llm_got == expected)

        llm_results.append({
            "id": p_id,
            "category": p.get("category", ""),
            "expected": expected,
            "got": llm_got,
            "confidence": llm_res.confidence,
            "reason": llm_res.reason,
            "verifier_type": llm_res.verifier_type,
            "matched": llm_matched
        })
        print(
            f"#{p_id:02d} | Exp: {expected:<15} | LLM Got: {llm_got:<15} ({llm_res.confidence:.2f}) | Match: {str(llm_matched):<5} | Reason: {llm_res.reason[:65]}...",
            flush=True
        )

        # 2. Run rule verifier for composite synthesis
        rule_res = rule_verifier.verify(p["claim"], p["evidence"])
        if rule_res.label in (NLILabel.SUPPORTED, NLILabel.CONTRADICTED) and rule_res.confidence >= 0.85:
            comp_got = rule_res.label.value
            comp_conf = rule_res.confidence
            comp_vtype = "rule"
            comp_reason = rule_res.reason
        else:
            comp_got = llm_got
            comp_conf = llm_res.confidence
            comp_vtype = "llm"
            comp_reason = llm_res.reason

        comp_matched = (comp_got == expected)
        composite_results.append({
            "id": p_id,
            "category": p.get("category", ""),
            "expected": expected,
            "got": comp_got,
            "confidence": comp_conf,
            "verifier_type": comp_vtype,
            "reason": comp_reason,
            "matched": comp_matched
        })

    # Summary metrics helper
    def compute_metrics(res_list: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        neg_count = 0
        false_positives = 0
        contra_count = 0
        contra_caught = 0
        by_label: Dict[str, Dict[str, int]] = {}

        for r in res_list:
            exp = r["expected"]
            got = r["got"]
            is_neg = (exp in ("NOT_SUPPORTED", "CONTRADICTED"))
            if is_neg:
                neg_count += 1
                if got == "SUPPORTED":
                    false_positives += 1

            if exp == "CONTRADICTED":
                contra_count += 1
                if got == "CONTRADICTED":
                    contra_caught += 1

            by_label.setdefault(exp, {"total": 0, "correct": 0})
            by_label[exp]["total"] += 1
            if r["matched"]:
                by_label[exp]["correct"] += 1

        fp_rate = (false_positives / neg_count) * 100 if neg_count else 0
        contra_recall = (contra_caught / contra_count) * 100 if contra_count else 0
        total_correct = sum(st["correct"] for st in by_label.values())
        overall_acc = (total_correct / len(res_list)) * 100 if res_list else 0

        print(f"\n--- {name} SUMMARY ---", flush=True)
        print(f"False Positive SUPPORTED rate on negatives: {false_positives}/{neg_count} ({fp_rate:.1f}%)", flush=True)
        print(f"CONTRADICTED recall: {contra_caught}/{contra_count} ({contra_recall:.1f}%)", flush=True)
        for lbl, st in sorted(by_label.items()):
            acc = (st["correct"] / st["total"]) * 100 if st["total"] else 0
            print(f"  Label {lbl:<15}: {st['correct']}/{st['total']} correct ({acc:.1f}%)", flush=True)
        print(f"Overall Accuracy: {total_correct}/{len(res_list)} ({overall_acc:.1f}%)", flush=True)

        return {
            "false_positives": false_positives,
            "negative_count": neg_count,
            "false_positive_rate": fp_rate,
            "contra_caught": contra_caught,
            "contra_count": contra_count,
            "contra_recall": contra_recall,
            "by_label": by_label,
            "overall_accuracy": overall_acc
        }

    llm_metrics = compute_metrics(llm_results, "STANDALONE LLM (SmolLM3)")

    print("\n==================================================================", flush=True)
    print(" 2. COMPOSITE SYSTEM EVALUATION (Rules Fast-Path + LLM Fallback)", flush=True)
    print("==================================================================", flush=True)
    rule_handled = sum(1 for r in composite_results if r["verifier_type"] == "rule")
    llm_handled = sum(1 for r in composite_results if r["verifier_type"] == "llm")
    print(f"Routing: Rule-handled = {rule_handled}/{len(pairs)} ({rule_handled/len(pairs)*100:.1f}%), LLM-handled = {llm_handled}/{len(pairs)} ({llm_handled/len(pairs)*100:.1f}%)\n", flush=True)

    for r in composite_results:
        p_id = r["id"]
        exp = r["expected"]
        got = r["got"]
        vtype = r["verifier_type"]
        matched = r["matched"]
        print(f"#{p_id:02d} | Exp: {exp:<15} | Comp Got: {got:<15} ({r['confidence']:.2f} by {vtype:<5}) | Match: {str(matched):<5}", flush=True)

    composite_metrics = compute_metrics(composite_results, "COMPOSITE SYSTEM (Rules + LLM)")

    return {
        "llm_results": llm_results,
        "llm_metrics": llm_metrics,
        "composite_results": composite_results,
        "composite_metrics": composite_metrics,
        # Backward compatibility fields
        "false_positive_rate": composite_metrics["false_positive_rate"],
        "overall_accuracy": composite_metrics["overall_accuracy"]
    }


if __name__ == "__main__":
    run_calibration()
