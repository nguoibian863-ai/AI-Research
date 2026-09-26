import re
from typing import List, Dict, Any, Set

GENERIC_RESEARCH_TERMS = {
    "the", "and", "for", "with", "from", "that", "this", "which", "what", "how",
    "why", "where", "when", "does", "did", "are", "were", "been", "has", "have",
    "had", "not", "but", "can", "could", "will", "would", "about", "into", "over",
    "after", "between", "versus", "vs",
    "compare", "comparison", "comparing", "eval", "evaluation", "evaluating", "evaluate",
    "benchmark", "benchmarks", "benchmarking", "performance", "overview", "review", "summary",
    "analysis", "study", "article", "report", "guide", "post", "blog", "paper", "papers",
    "test", "tests", "testing",
    "speedup", "speed", "accuracy", "latency", "throughput", "fps", "map", "nds",
    "metric", "metrics", "result", "results", "score", "scores", "rate", "rates",
    "cost", "overhead", "efficiency", "scalability", "size", "parameter", "parameters"
}


def extract_core_entities(text: str) -> List[str]:
    """
    Extracts substantive core domain entities (e.g. model names, datasets, systems)
    from goal or query text, filtering out generic research terms and stop words.
    Preserves order of appearance.
    """
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
    entities: List[str] = []
    for t in raw_tokens:
        if t not in GENERIC_RESEARCH_TERMS and len(t) >= 3:
            if t not in entities:
                entities.append(t)

    if not entities:
        # Fallback for minimal queries consisting solely of generic terms (e.g. minimal unit tests)
        basic_stops = {
            "the", "and", "for", "with", "from", "that", "this", "which", "what",
            "how", "why", "where", "when", "does", "did", "are", "were", "been"
        }
        for t in raw_tokens:
            if t not in basic_stops and len(t) >= 2:
                if t not in entities:
                    entities.append(t)
    return entities


def check_entity_in_text(entity: str, text: str) -> bool:
    """
    Whole-word matching for a specific entity in text.
    Handles hyphenated entities (e.g. 'rt-detr') cleanly without substring false positives.
    """
    if not entity or not text:
        return False
    pattern = rf"(?<![a-zA-Z0-9_-]){re.escape(entity.lower())}(?![a-zA-Z0-9_-])"
    return bool(re.search(pattern, text.lower()))


def evaluate_coverage(entities: List[str], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates whether each required entity is present in at least one chunk.
    Returns:
    {
        "covered": List[str],
        "missing": List[str],
        "coverage_ratio": float,
        "is_complete": bool
    }
    """
    if not entities:
        return {
            "covered": [],
            "missing": [],
            "coverage_ratio": 1.0,
            "is_complete": True
        }

    covered = []
    missing = []
    for ent in entities:
        has_chunk = any(check_entity_in_text(ent, c.get("text", "")) for c in chunks)
        if has_chunk:
            covered.append(ent)
        else:
            missing.append(ent)

    ratio = len(covered) / len(entities) if entities else 1.0
    return {
        "covered": covered,
        "missing": missing,
        "coverage_ratio": ratio,
        "is_complete": (len(missing) == 0)
    }
