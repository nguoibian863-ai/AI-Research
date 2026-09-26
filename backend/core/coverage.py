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
    "achieve", "achieves", "achieved", "achieving",
    "show", "shows", "shown", "showing",
    "yield", "yields", "yielded", "yielding",
    "reach", "reaches", "reached", "reaching",
    "obtain", "obtains", "obtained", "obtaining",
    "demonstrate", "demonstrates", "demonstrated", "demonstrating",
    "present", "presents", "presented", "presenting",
    "propose", "proposes", "proposed", "proposing",
    "introduce", "introduces", "introduced", "introducing",
    "report", "reports", "reported", "reporting",
    "exceed", "exceeds", "exceeded", "exceeding",
    "surpass", "surpasses", "surpassed", "surpassing",
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
    "verify", "verifies", "verified", "verifying", "verification",
    "claim", "claims", "claimed", "claiming",
    "fact", "facts", "factual",
    "check", "checks", "checked", "checking",
    "investigate", "investigates", "investigated", "investigating", "investigation",
    "assess", "assesses", "assessed", "assessing", "assessment",
    "validate", "validates", "validated", "validating", "validation",
    "find", "finds", "found", "finding", "findings",
    "detail", "details", "detailed",
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
        if t in GENERIC_RESEARCH_TERMS or t.isdigit():
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


DATASET_BENCHMARK_TERMS = {
    # Computer Vision
    "nuscenes", "coco", "kitti", "imagenet", "waymo", "cityscapes", "voc",
    "pascal", "glue", "squad", "cifar", "cifar10", "cifar100", "mnist",
    "lvis", "wider", "mot17", "mot20", "bdd100k", "argoverse", "shapenet",
    "scanrefer", "scannet", "sunrgbd", "tum", "kitti360",
    # Database / Systems
    "tpcc", "tpc-c", "tpch", "tpc-h", "ycsb", "sysbench", "chbenchmark",
    "ch-benchmark", "linkbench", "graph500", "db_bench", "dbbench",
    # LLM / NLP
    "mmlu", "gsm8k", "humaneval", "arc", "hellaswag", "math", "winogrande",
    "drop", "truthfulqa", "alpacaeval", "mt-bench", "mtbench", "swe-bench",
    "swebench", "superglue",
    # CPU / Hardware / Compute
    "spec", "specint", "specfp", "specrate", "coremark", "cinebench",
    "geekbench", "mlperf", "linpack", "stream"
}


def extract_context_entities(text: str) -> Set[str]:
    """
    Extracts benchmark, dataset, or environmental context entities that follow
    prepositions like 'on', 'in', 'using', 'across', 'under', 'with' in research goals.
    E.g. in 'Compare Postgres and MySQL on TPC-C', 'tpc-c' is a context entity.
    Also handles conjunctions like 'on MMLU and GSM8K'.
    """
    if not text:
        return set()
    context_ents = set()
    # Match clause after prepositions up to punctuation or end of string
    matches = re.findall(
        r"\b(?:on|in|using|across|under|with)\s+([^,.;:!?\n]+)",
        text,
        re.IGNORECASE
    )
    for m in matches:
        sub_tokens = extract_core_entities(m)
        for tok in sub_tokens:
            context_ents.add(tok.lower())
    return context_ents


METRIC_UNITS_RE = re.compile(
    r"^(?:\s*(?:%|ms|s|fps|map|nds|ap|gb|mb|kb|ghz|mhz|tpmc|ops/s|qps|queries/s|tokens/s|tok/s|times|x|parameters|param|layers|b|m|k))\b",
    re.IGNORECASE
)


def extract_substantive_numbers(text: str) -> List[str]:
    """
    Extracts substantive numeric metrics from sentence text while ignoring:
    1. Citation brackets like [E1], [E2]
    2. URLs (e.g. arxiv IDs in https://...)
    3. Model/hardware identifiers (e.g. 3D, 2D, YOLOv8, GPT-4, A100)
    4. 4-digit calendar years (1900-2099) without metric units
    5. Standalone small integer counts (1-5) without metric units (e.g. '2 models', '3 stages')
    """
    if not text:
        return []

    # 1. Strip citations and URLs
    clean_text = re.sub(r"\[E\d+\]", " ", text)
    clean_text = re.sub(r"https?://\S+", " ", clean_text)

    # 2. Mask out model / hardware / dimension tokens like 3D, 2D, YOLOv8, A100, GPT-4
    clean_text = re.sub(r"\b\d+[dD]\b", " ", clean_text)
    clean_text = re.sub(r"\b[a-zA-Z]+[-_]?\d+[a-zA-Z0-9_-]*\b", " ", clean_text)

    substantive = []
    for m in re.finditer(r"\b\d+(?:\.\d+)?\b", clean_text):
        num_str = m.group(0)

        # Floating point decimals (e.g. 60.3, 0.99, 95.0, 10.0) are always substantive
        if "." in num_str:
            substantive.append(num_str)
            continue

        val = int(num_str)
        trailing_text = clean_text[m.end():m.end() + 20]
        has_metric_unit = bool(METRIC_UNITS_RE.search(trailing_text))

        # Ignore 4-digit calendar years (e.g. 2019, 2024) unless directly attached to a metric unit
        if 1900 <= val <= 2099 and not has_metric_unit:
            continue

        # Ignore small integer counts (1-5) if they have no metric units (e.g. '2 models', '3 steps')
        if 1 <= val <= 5 and not has_metric_unit:
            continue

        substantive.append(num_str)

    return substantive


def evaluate_evidence_coverage(
    entities: List[str],
    evidence_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates whether each required entity is supported by at least one verified atomic evidence item.
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

    covered = []
    for ent in entities:
        ent_found = False
        for ev in evidence_items:
            ev_text = f"{ev.get('subject', '')} {ev.get('predicate', '')} {ev.get('statement', '')} {ev.get('exact_quote', '')} {ev.get('raw_quote', '')}"
            if check_entity_in_text(ent, ev_text):
                ent_found = True
                break
        if ent_found:
            covered.append(ent)

    missing = [e for e in entities if e not in covered]
    ratio = len(covered) / len(entities) if entities else 1.0
    return {
        "covered": covered,
        "missing": missing,
        "coverage_ratio": ratio,
        "is_complete": len(missing) == 0
    }
