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
    "cost", "overhead", "efficiency", "scalability", "size", "parameter", "parameters",
    "detection", "detecting", "detector", "classification", "segmentation", "recognition", "tracking",
    "3d", "2d",
    "affect", "affects", "affecting", "effect", "effects", "effective",
    "use", "using", "uses", "used", "instead", "tradeoff", "tradeoffs",
    "service", "services", "backend", "backends", "frontend", "frontends",
    "difference", "differences", "different", "impact", "impacts", "impacted",
    "role", "roles", "better", "best", "worse", "worst", "pros", "cons",
    "advantage", "advantages", "disadvantage", "disadvantages",
    "recommend", "recommendation", "recommendations",
    "implement", "implementation", "implementations",
    "approach", "approaches", "method", "methods", "technique", "techniques",
    "work", "works", "working", "consumer", "enterprise", "choice", "choose",
    "choosing", "various", "multiple", "several", "general", "common",
    "like", "such", "than", "more", "most", "less", "least",
    "of", "to", "in", "on", "at", "by", "as", "is", "it", "an", "or", "we",
    "do", "so", "no", "if", "my", "up", "out", "off", "all", "any", "some",
    "every", "each", "both", "neither", "either", "one", "two", "first", "second", "last"
}


def extract_core_entities(text: str) -> List[str]:
    """
    Extracts substantive core domain entities (e.g. model names, datasets, systems, languages)
    from goal or query text, filtering out generic research terms and stop words.
    Preserves 2-letter capitalized names or terms with digits (e.g. 'Go', 'AI', 'C#', 'R', 'YOLOv8').
    Preserves order of appearance.
    """
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text)
    entities: List[str] = []

    for orig_t in raw_tokens:
        t = orig_t.lower()
        if t in GENERIC_RESEARCH_TERMS:
            continue

        if len(t) == 2:
            # Keep 2-character tokens if capitalized (e.g. 'Go', 'AI', 'ML') or containing digits/known terms
            if orig_t[0].isupper() or any(c.isdigit() for c in orig_t) or t in {"go", "ai", "ml", "db", "os", "ip", "c#"}:
                if t not in entities:
                    entities.append(t)
        elif len(t) >= 3:
            if t not in entities:
                entities.append(t)

    if not entities:
        # Fallback for minimal queries consisting solely of generic terms
        basic_stops = {
            "the", "and", "for", "with", "from", "that", "this", "which", "what",
            "how", "why", "where", "when", "does", "did", "are", "were", "been"
        }
        for orig_t in raw_tokens:
            t = orig_t.lower()
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
        "covered": [str, ...],
        "missing": [str, ...],
        "coverage_ratio": float,
        "is_complete": bool
    }
    """
    if not entities:
        return {"covered": [], "missing": [], "coverage_ratio": 1.0, "is_complete": True}

    chunks_text = "\n".join(c.get("text", "") for c in chunks)
    covered = [e for e in entities if check_entity_in_text(e, chunks_text)]
    missing = [e for e in entities if e not in covered]

    coverage_ratio = len(covered) / len(entities) if entities else 1.0
    return {
        "covered": covered,
        "missing": missing,
        "coverage_ratio": coverage_ratio,
        "is_complete": len(missing) == 0
    }
