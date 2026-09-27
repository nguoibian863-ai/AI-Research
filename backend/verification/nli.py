import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Set

from backend.verification.schemas import NLILabel, NLIVerificationResult, NLIStructuredOutputSchema
from backend.llm.backend import LLMBackend
from backend.core.coverage import (
    extract_substantive_numbers,
    subject_matches_entity,
    extract_core_entities,
    DATASET_BENCHMARK_TERMS,
)

logger = logging.getLogger(__name__)

NLI_PROMPT_VERSION = "v1.1"

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could",
    "will", "would", "shall", "should", "it", "its", "that", "this", "these", "those",
    "which", "who", "whom", "what", "as", "if", "so", "too", "very", "our", "their", "of"
}

HARDWARE_ENV_TERMS = {
    "t4", "gpu", "gpus", "v100", "a100", "h100", "cpu", "cpus", "cuda",
    "tensorrt", "trt", "fp16", "fp32", "int8", "bf16", "fps", "latency",
    "throughput", "ap", "map", "nds", "ms"
}

LINKING_VERBS = {
    "is", "are", "was", "were", "be", "been", "being",
    "run", "runs", "ran", "running",
    "achieve", "achieves", "achieved", "achieving",
    "perform", "performs", "performed", "performing",
    "yield", "yields", "yielded", "yielding",
    "reach", "reaches", "reached", "reaching",
    "show", "shows", "shown", "showing",
    "operate", "operates", "operated", "operating",
    "use", "uses", "used", "using",
    "appear", "appears", "appeared", "appearing",
    "seem", "seems", "seemed", "seeming",
    "become", "becomes", "became", "becoming",
}

GENERIC_DOMAIN_NOUNS: Set[str] = {
    "detector", "detectors", "model", "models", "system", "systems",
    "network", "networks", "architecture", "architectures", "method", "methods",
    "approach", "approaches", "baseline", "baselines", "algorithm", "algorithms",
    "framework", "frameworks", "technique", "techniques", "solution", "solutions",
    "pipeline", "pipelines", "variant", "variants", "family", "families",
    "real-time", "realtime", "sota", "state-of-the-art"
}

CONTEXT_EXCLUDED = DATASET_BENCHMARK_TERMS | HARDWARE_ENV_TERMS | LINKING_VERBS | GENERIC_DOMAIN_NOUNS

# Metric polarities (Plan 23 & Review items 3)
LOWER_IS_BETTER_METRICS: Set[str] = {
    "latency", "error", "error rate", "loss", "runtime", "time", "ms",
    "params", "parameters", "flops", "memory", "footprint", "vram", "cost", "delay",
    "training time", "train time", "inference time", "size", "model size"
}
HIGHER_IS_BETTER_METRICS: Set[str] = {
    "accuracy", "acc", "ap", "map", "nds", "throughput", "fps", "score",
    "f1", "precision", "recall", "speedup", "tps", "speed", "efficiency",
    "inference speed", "sample efficiency"
}
ALL_KNOWN_METRICS: Set[str] = LOWER_IS_BETTER_METRICS | HIGHER_IS_BETTER_METRICS

SPEED_METRIC_TERMS: Set[str] = {
    "latency", "speed", "fps", "ms", "throughput", "runtime", "time", "inference",
    "inference time", "inference latency", "inference speed", "delay", "speedup",
    "tps", "training time", "train time"
}
ACCURACY_METRIC_TERMS: Set[str] = {
    "accuracy", "acc", "ap", "map", "nds", "score", "f1", "precision", "recall",
    "error", "error rate", "loss"
}

# Direct opposing comparative pairs that inherently specify dimension
INHERENT_OPPOSING_PAIRS: Dict[str, str] = {
    "faster": "slower",
    "slower": "faster",
    "better": "worse",
    "worse": "better",
    "superior": "inferior",
    "inferior": "superior",
    "outperforms": "underperforms",
    "underperforms": "outperforms",
    "exceeds": "trails",
    "trails": "exceeds",
    "beats": "loses to",
    "loses to": "beats",
}

# Generic relative comparatives whose direction meaning depends on the metric
GENERIC_RELATIVE_COMPARATIVES: Set[str] = {
    "higher", "lower", "more", "less", "greater", "smaller", "larger"
}

# All comparative pairs
OPPOSING_PAIRS: Dict[str, str] = {
    **INHERENT_OPPOSING_PAIRS,
    "higher": "lower",
    "lower": "higher",
    "more": "less",
    "less": "more",
    "greater": "smaller",
    "smaller": "greater",
    "larger": "smaller",
}
SUPERIOR_TERMS = {
    "faster", "higher", "more", "better", "superior",
    "outperforms", "exceeds", "beats", "greater", "larger"
}
INFERIOR_TERMS = {v for k, v in OPPOSING_PAIRS.items() if k in SUPERIOR_TERMS}
ALL_COMPARATIVES: Set[str] = set(OPPOSING_PAIRS.keys())

# Explicit verbal negation markers
VERBAL_NEGATIONS: Set[str] = {
    "not", "never", "cannot", "cant", "fails to", "unable to", "neither", "nor"
}


def _is_number_or_unit(token: str) -> bool:
    """Detects whether a token is a number, multiplier, or metric unit."""
    t = token.lower().strip()
    if not t:
        return False
    # Pure number: 1.5, 10, 59.2
    if re.fullmatch(r"\d+(\.\d+)?", t):
        return True
    # Multiplier: 2x, 10x, 1.5x
    if re.fullmatch(r"\d+(\.\d+)?x", t):
        return True
    # Units and symbols
    if t in {
        "ms", "fps", "ap", "map", "nds", "tps", "flops", "gflops", "tflops",
        "mb", "gb", "kb", "pts", "%", "percent", "points"
    }:
        return True
    return False


def _extract_metric_terms(text: str) -> Set[str]:
    """Finds known metric words in text."""
    lower_t = text.lower()
    found = set()
    for m in ALL_KNOWN_METRICS:
        if re.search(rf"\b{re.escape(m)}\b", lower_t):
            found.add(m)
    return found


def _get_metric_polarity(comp: str, metric: str) -> int:
    """
    Returns:
      +1 if the comparative indicates an advantage/superiority on this metric,
      -1 if the comparative indicates an inferiority/drawback on this metric,
       0 if neutral, unknown, or not metric-specific.
    """
    c = comp.lower().strip()
    m = metric.lower().strip()

    if c in {"faster", "better", "superior", "outperforms", "exceeds", "beats"}:
        return 1
    if c in {"slower", "worse", "inferior", "underperforms", "trails", "loses to"}:
        return -1

    if m in LOWER_IS_BETTER_METRICS:
        if c in {"lower", "less", "smaller", "fewer"}:
            return 1
        if c in {"higher", "more", "greater", "larger"}:
            return -1
    elif m in HIGHER_IS_BETTER_METRICS:
        if c in {"higher", "more", "greater", "larger"}:
            return 1
        if c in {"lower", "less", "smaller", "fewer"}:
            return -1

    return 0


def _resolve_polarity_and_domain(comp: str, metrics: Set[str]) -> Tuple[int, str]:
    """
    Resolves the semantic polarity (+1 advantage, -1 disadvantage) and domain (speed, accuracy, general)
    of a comparative relation given the comparative word and context metrics.
    """
    c = comp.lower().strip()

    # 1. Determine domain from context metrics if available
    domain = "general"
    for m in metrics:
        if m in SPEED_METRIC_TERMS:
            domain = "speed"
            break
        elif m in ACCURACY_METRIC_TERMS:
            domain = "accuracy"
            break

    # 2. Check metric-specific polarity
    for m in metrics:
        pol = _get_metric_polarity(c, m)
        if pol != 0:
            return pol, domain

    # 3. Check inherent polarity of comparative word
    if c in {"faster"}:
        return 1, "speed"
    if c in {"slower"}:
        return -1, "speed"
    if c in {"better", "superior", "outperforms", "exceeds", "beats"}:
        return 1, domain
    if c in {"worse", "inferior", "underperforms", "trails", "loses to"}:
        return -1, domain

    # 4. Fallback for generic relative comparatives without metric (higher, lower)
    pol = _get_metric_polarity(c, "")
    if pol != 0:
        return pol, domain

    return 0, domain if domain != "general" else "unknown"


def strip_citation_markers(text: str) -> str:
    """Removes [E1], **E1**, (E1), E1 citation tags from sentence."""
    t = re.sub(r"\[E\d+\]", " ", text)
    t = re.sub(r"\*\*E\d+\*\*", " ", t)
    t = re.sub(r"\(E\d+\)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def clean_claim_for_nli(claim: str) -> str:
    """Removes citation markers and attribution phrases like 'According to [E1], ' so NLI only evaluates the factual assertion."""
    t = re.sub(r"^(?:according\s+to|as\s+stated\s+in|as\s+reported\s+in|per|in)\s+\[?E\d+\]?[\s,:]*", "", claim, flags=re.IGNORECASE)
    t = re.sub(r"^\[?E\d+\]?\s+(?:reports|states|shows|indicates|mentions|demonstrates)\s+that\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\[E\d+\]", "", t)
    t = re.sub(r"\*\*E\d+\*\*", "", t)
    t = re.sub(r"\(E\d+\)", "", t)
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t


def _normalize_text_for_match(text: str) -> str:
    """Keep decimals like 67.3 intact, strip punctuation and citation brackets, lowercase and whitespace-normalize."""
    t = strip_citation_markers(text).lower()
    t = re.sub(r"(?<=\d)\.(?=\d)", "__DOT__", t)
    t = re.sub(r"[^a-zA-Z0-9_\-\s]", " ", t)
    t = t.replace("__DOT__", ".")
    return re.sub(r"\s+", " ", t).strip()


VARIANT_SUFFIX_PATTERN = re.compile(
    r"^(?:[nsmlx]|nano|tiny|small|base|large|xlarge|huge|r\d+|d\d+|b\d+|v\d+|\d+)$",
    re.IGNORECASE
)


def _extract_base_and_suffix(entity: str) -> Tuple[str, str]:
    """
    Separates a model entity into its base family and variant suffix.
    E.g.:
      'yolov8-l' -> ('yolov8', 'l')
      'yolov8_n' -> ('yolov8', 'n')
      'yolov8n'  -> ('yolov8', 'n')
      'rt-detr-r50' -> ('rt-detr', 'r50')
      'rt-detr'  -> ('rt-detr', '')
      'yolov8'   -> ('yolov8', '')
      'resnet50' -> ('resnet', '50')
    """
    ent = entity.lower().strip()
    # 1. Hyphen or underscore followed by recognized variant: e.g. -l, -r50, -nano
    m = re.match(r"^([a-z0-9_\-\.]+)[-_]([a-z0-9]+)$", ent)
    if m and VARIANT_SUFFIX_PATTERN.match(m.group(2)):
        return m.group(1), m.group(2)

    # 2. Attached letter variant for common models: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    m2 = re.match(r"^(yolo[v\d]+|rt-?detr|resnet\d*|vit[-_]?[a-z\d]*)([nsmlx])$", ent)
    if m2:
        return m2.group(1), m2.group(2)

    # 3. Model family followed by numeric variant: resnet50, resnet101, vgg16, etc.
    m3 = re.match(r"^(resnet|vgg|densenet|efficientnet[-_]?[b\d]*|mobilenet[-_]?[v\d]*)[-_]?(\d+)$", ent)
    if m3:
        return m3.group(1), m3.group(2)

    return ent, ""


def _entities_match(e1: str, e2: str) -> bool:
    """Checks whether two entities are identical or compatible variants of the same model/subject."""
    if not e1 or not e2:
        return False
    e1_l, e2_l = e1.lower().strip(), e2.lower().strip()
    if e1_l == e2_l:
        return True

    b1, s1 = _extract_base_and_suffix(e1_l)
    b2, s2 = _extract_base_and_suffix(e2_l)

    # If both have different non-empty variant suffixes, they are DIFFERENT model variants!
    if s1 and s2 and s1 != s2:
        return False

    return subject_matches_entity(e1_l, e2_l) or subject_matches_entity(e2_l, e1_l)


def _clean_token(t: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "", t.lower())


def _tokenize_content_words(text: str) -> List[str]:
    """Extract substantive lowercased alphanumeric content words excluding common stopwords."""
    words = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def extract_comparative_triples(text: str, known_entities: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
    """
    Extracts structured comparative relations from text: (subject, comparative_word, object).
    Uses strict word boundaries, clause-aware comparative scoping, and prioritizes known research goal entities.
    """
    clean_text = strip_citation_markers(text).lower()
    triples: List[Tuple[str, str, str]] = []

    # 1. Pattern: <EntityA> ... <comp_word> ... than <EntityB>
    than_matches = list(re.finditer(r"\bthan\s+([a-zA-Z0-9_\-\.]+)\b", clean_text))
    all_than_objects = {tm.group(1).strip() for tm in than_matches}

    prev_than_end = -1
    for tm in than_matches:
        obj_entity = tm.group(1).strip()
        than_start = tm.start()

        if obj_entity in STOPWORDS or obj_entity in CONTEXT_EXCLUDED or _is_number_or_unit(obj_entity):
            prev_than_end = tm.end()
            continue

        before_than = clean_text[:than_start]

        # Search for comparative words using word boundaries
        # For this 'than', only look for comparative words that appear AFTER the previous 'than'!
        clause_comps: List[Tuple[int, int, str]] = []
        for comp in ALL_COMPARATIVES:
            for cm in re.finditer(rf"\b{re.escape(comp)}\b", before_than):
                if cm.start() > prev_than_end:
                    clause_comps.append((cm.start(), cm.end(), comp))

        prev_than_end = tm.end()
        if not clause_comps:
            continue

        # Per review guidelines: For each 'than', pick the CLOSEST comparative word preceding it
        clause_comps.sort(key=lambda x: x[0])
        c_start, c_end, comp = clause_comps[-1]  # Closest comparative word before 'than'

        text_before_comp = before_than[:c_start].strip()
        tokens_before = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text_before_comp)

        # Search backwards from comparative word to find closest substantive subject
        sub_entity = None

        # Priority 1: Check if any token matches known research goal entities
        # Skip tokens that are objects of 'than' clauses or comparative words to avoid picking previous clause elements as subject
        if known_entities:
            for tok in reversed(tokens_before):
                if tok in all_than_objects or tok in ALL_COMPARATIVES:
                    continue
                if any(_entities_match(tok, ke) for ke in known_entities):
                    sub_entity = tok
                    break

        # Priority 2: Substantive token excluding stopwords, numbers, hardware, generic domain nouns, comparatives, and than-objects
        if not sub_entity:
            for tok in reversed(tokens_before):
                if tok in all_than_objects or tok in ALL_COMPARATIVES:
                    continue
                if (
                    tok not in STOPWORDS
                    and tok not in CONTEXT_EXCLUDED
                    and not _is_number_or_unit(tok)
                    and len(tok) > 1
                ):
                    sub_entity = tok
                    break

        if not sub_entity:
            for tok in reversed(tokens_before):
                if tok in all_than_objects or tok in ALL_COMPARATIVES:
                    continue
                if (
                    tok not in STOPWORDS
                    and not _is_number_or_unit(tok)
                    and len(tok) > 1
                ):
                    sub_entity = tok
                    break

        if sub_entity and not _entities_match(sub_entity, obj_entity):
            triples.append((sub_entity, comp, obj_entity))

    # 2. Pattern: <EntityA> (outperforms|underperforms|beats|trails|exceeds) <EntityB>
    verb_pattern = r"\b([a-zA-Z0-9_\-\.]+)\b\s+(?:[a-zA-Z0-9_\-\.]+\s+){0,2}?\b(outperforms|underperforms|beats|trails|exceeds)\b\s+(?:(?:the|all|prior|other|both|existing|baseline)\s+)*\b([a-zA-Z0-9_\-\.]+)\b"
    for m in re.finditer(verb_pattern, clean_text):
        sub, comp, obj = m.group(1), m.group(2), m.group(3)
        if (
            sub not in STOPWORDS
            and obj not in STOPWORDS
            and sub not in CONTEXT_EXCLUDED
            and obj not in CONTEXT_EXCLUDED
            and not _entities_match(sub, obj)
        ):
            triples.append((sub, comp, obj))

    return triples


def _get_comp_direction(comp: str) -> int:
    """Returns +1 for superior/outperforming comparatives, -1 for inferior/underperforming."""
    if comp in SUPERIOR_TERMS:
        return 1
    if comp in INFERIOR_TERMS:
        return -1
    return 0


class BaseNLIVerifier(ABC):
    """Abstract base class for Natural Language Inference / Entailment Verifiers (Plan 23)."""

    @abstractmethod
    def verify(
        self,
        claim: str,
        evidence: str,
        budget_tracker: Optional[Any] = None,
        all_quotes: Optional[List[str]] = None,
        known_entities: Optional[List[str]] = None
    ) -> NLIVerificationResult:
        """Verify entailment between claim and evidence."""
        pass

    def verify_batch(self, pairs: List[Tuple[str, str]], budget_tracker: Optional[Any] = None) -> List[NLIVerificationResult]:
        """Verify multiple claim-evidence pairs."""
        return [self.verify(claim, evidence, budget_tracker=budget_tracker) for claim, evidence in pairs]


class RuleBasedNLIVerifier(BaseNLIVerifier):
    """
    High-precision, zero-latency rule-based entailment checker.
    Specifically checks:
    1. Near-identity / substring match (only source of rule-based SUPPORTED with confidence 1.0)
    2. Verbal negation guardrails
    3. Entity-aware comparative relation and direction consistency (detects CONTRADICTED)
    4. Numeric consistency check
    5. Token overlap (capped at PARTIALLY_SUPPORTED with confidence 0.70 to never bypass LLM)
    """

    def verify(
        self,
        claim: str,
        evidence: str,
        budget_tracker: Optional[Any] = None,
        all_quotes: Optional[List[str]] = None,
        known_entities: Optional[List[str]] = None
    ) -> NLIVerificationResult:
        c_clean = claim.strip()
        e_clean = evidence.strip()

        if not c_clean or not e_clean:
            return NLIVerificationResult(
                label=NLILabel.NOT_SUPPORTED,
                confidence=1.0,
                reason="Empty claim or evidence text",
                verifier_type="rule_based",
                claim_text=c_clean,
                evidence_quote=e_clean
            )

        c_stripped = strip_citation_markers(c_clean)
        c_norm = _normalize_text_for_match(c_clean)
        e_norm = _normalize_text_for_match(e_clean)

        # 1. Exact or near-exact normalized match (P0 golden rule for SUPPORTED)
        # ONLY accept if claim is identical to or fully contained within evidence quote!
        # Do NOT accept e_norm in c_norm, which allows hallucinated extensions to pass.
        if c_norm == e_norm or c_norm in e_norm:
            return NLIVerificationResult(
                label=NLILabel.SUPPORTED,
                confidence=1.0,
                reason="Claim is an exact or direct substring match of evidence quote",
                verifier_type="rule_based",
                claim_text=c_clean,
                evidence_quote=e_clean
            )

        # 2. Verbal Negation Guardrail (Review 485345a item 1)
        c_has_neg = any(re.search(rf"\b{re.escape(neg)}\b", c_clean.lower()) for neg in VERBAL_NEGATIONS)
        e_has_neg = any(re.search(rf"\b{re.escape(neg)}\b", e_clean.lower()) for neg in VERBAL_NEGATIONS)
        if c_has_neg != e_has_neg:
            return NLIVerificationResult(
                label=NLILabel.NOT_SUPPORTED,
                confidence=0.50,
                reason="Verbal negation mismatch between claim and evidence, deferring to LLM verifier",
                verifier_type="rule_based",
                claim_text=c_clean,
                evidence_quote=e_clean
            )

        # 3. Entity-aware comparative relation check with Metric Polarity awareness
        c_triples = extract_comparative_triples(c_clean, known_entities=known_entities)
        e_triples = extract_comparative_triples(e_clean, known_entities=known_entities)

        for c_sub, c_comp, c_obj in c_triples:
            c_metrics = _extract_metric_terms(c_clean)
            c_pol, c_dom = _resolve_polarity_and_domain(c_comp, c_metrics)

            # Check if this triple is already explicitly supported by evidence
            has_agreeing = False
            for e_s, e_c, e_o in e_triples:
                e_metrics = _extract_metric_terms(e_clean)
                shared_metrics = c_metrics & e_metrics
                e_pol, e_dom = _resolve_polarity_and_domain(e_c, e_metrics)

                same_specific_domain = (c_dom == e_dom and c_dom in ("speed", "accuracy"))
                is_metric_comparable = (
                    same_specific_domain
                    or bool(shared_metrics)
                    or (not c_metrics and not e_metrics and c_dom in ("general", "unknown") and e_dom in ("general", "unknown"))
                )

                # Direct match: same subject and same object
                if _entities_match(c_sub, e_s) and _entities_match(c_obj, e_o):
                    if c_comp == e_c and (shared_metrics or not (c_metrics or e_metrics)):
                        has_agreeing = True
                        break
                    if is_metric_comparable and c_pol != 0 and e_pol != 0 and c_pol == e_pol:
                        has_agreeing = True
                        break

                # Swapped match with opposite polarity: "A is slower than B" agrees with "B is faster than A"
                if _entities_match(c_sub, e_o) and _entities_match(c_obj, e_s):
                    if is_metric_comparable and c_pol != 0 and e_pol != 0 and c_pol == -e_pol:
                        has_agreeing = True
                        break

            if has_agreeing:
                continue

            for e_sub, e_comp, e_obj in e_triples:
                e_metrics = _extract_metric_terms(e_clean)
                shared_metrics = c_metrics & e_metrics

                c_pol, c_dom = _resolve_polarity_and_domain(c_comp, c_metrics)
                e_pol, e_dom = _resolve_polarity_and_domain(e_comp, e_metrics)

                # Check if domains directly clash (e.g. speed vs accuracy)
                cross_domain_clash = (
                    (c_dom == "speed" and e_dom == "accuracy")
                    or (c_dom == "accuracy" and e_dom == "speed")
                )
                if cross_domain_clash:
                    continue

                # 3a. Reversed comparison direction (e.g. "A is faster than B" vs "B is faster than A")
                if _entities_match(c_sub, e_obj) and _entities_match(c_obj, e_sub):
                    # P0 Guardrail: For swapped entities with inherent comparatives:
                    # If metrics differ or only one side specifies a metric, do NOT conclude CONTRADICTED! Defer to LLM.
                    same_specific_domain = (c_dom == e_dom and c_dom in ("speed", "accuracy"))
                    is_swapped_comparable = (
                        same_specific_domain
                        or bool(shared_metrics)
                        or (not c_metrics and not e_metrics)
                    )
                    if not is_swapped_comparable:
                        continue

                    if c_pol != 0 and e_pol != 0 and c_pol == e_pol:
                        return NLIVerificationResult(
                            label=NLILabel.CONTRADICTED,
                            confidence=0.95,
                            reason=f"Reversed comparison direction between {c_sub} and {c_obj}: claim asserts '{c_sub} {c_comp} {c_obj}' ({c_dom}), but evidence asserts '{e_sub} {e_comp} {e_obj}' ({e_dom})",
                            verifier_type="rule_based",
                            claim_text=c_clean,
                            evidence_quote=e_clean
                        )
                    elif c_comp == e_comp:
                        return NLIVerificationResult(
                            label=NLILabel.CONTRADICTED,
                            confidence=0.95,
                            reason=f"Reversed comparison direction between {c_sub} and {c_obj}: claim states '{c_sub} {c_comp} {c_obj}', but evidence states '{e_sub} {e_comp} {e_obj}'",
                            verifier_type="rule_based",
                            claim_text=c_clean,
                            evidence_quote=e_clean
                        )
                    elif c_comp in SUPERIOR_TERMS and e_comp in SUPERIOR_TERMS:
                        return NLIVerificationResult(
                            label=NLILabel.CONTRADICTED,
                            confidence=0.95,
                            reason=f"Reversed superior comparison between {c_sub} and {c_obj}: claim states '{c_sub} {c_comp} {c_obj}', but evidence states '{e_sub} {e_comp} {e_obj}'",
                            verifier_type="rule_based",
                            claim_text=c_clean,
                            evidence_quote=e_clean
                        )
                    elif c_comp in INFERIOR_TERMS and e_comp in INFERIOR_TERMS:
                        return NLIVerificationResult(
                            label=NLILabel.CONTRADICTED,
                            confidence=0.95,
                            reason=f"Reversed inferior comparison between {c_sub} and {c_obj}: claim states '{c_sub} {c_comp} {c_obj}', but evidence states '{e_sub} {e_comp} {e_obj}'",
                            verifier_type="rule_based",
                            claim_text=c_clean,
                            evidence_quote=e_clean
                        )

                # 3b. Opposite comparative polarity on same entity pair (e.g. "faster" vs "slower", "exceeds" vs "trails")
                if _entities_match(c_sub, e_sub) and _entities_match(c_obj, e_obj):
                    if c_metrics and e_metrics and not shared_metrics and (c_dom != e_dom):
                        continue

                    if c_pol != 0 and e_pol != 0 and c_pol != e_pol:
                        return NLIVerificationResult(
                            label=NLILabel.CONTRADICTED,
                            confidence=0.95,
                            reason=f"Opposite comparative polarity for ({c_sub} vs {c_obj}): claim asserts '{c_comp}' ({c_dom}) but evidence asserts '{e_comp}' ({e_dom})",
                            verifier_type="rule_based",
                            claim_text=c_clean,
                            evidence_quote=e_clean
                        )

                    # Case I: Inherent metric-specific opposing pairs (faster vs slower, outperforms vs underperforms, exceeds vs trails)
                    if INHERENT_OPPOSING_PAIRS.get(c_comp) == e_comp:
                        return NLIVerificationResult(
                            label=NLILabel.CONTRADICTED,
                            confidence=0.95,
                            reason=f"Opposite comparative direction for ({c_sub} vs {c_obj}): claim states '{c_comp}' but evidence states '{e_comp}'",
                            verifier_type="rule_based",
                            claim_text=c_clean,
                            evidence_quote=e_clean
                        )

                    # Case II: Generic relative comparatives (higher vs lower, more vs less)
                    # ONLY contradicts if comparing the EXACT SAME METRIC!
                    if OPPOSING_PAIRS.get(c_comp) == e_comp:
                        if shared_metrics:
                            return NLIVerificationResult(
                                label=NLILabel.CONTRADICTED,
                                confidence=0.95,
                                reason=f"Opposite comparative direction on metric '{next(iter(shared_metrics))}' for ({c_sub} vs {c_obj}): claim states '{c_comp}' but evidence states '{e_comp}'",
                                verifier_type="rule_based",
                                claim_text=c_clean,
                                evidence_quote=e_clean
                            )
                        # Without shared metric, do not declare CONTRADICTED; defer to LLM!

        # 4. Numeric consistency check
        # When all_quotes is provided (multi-citation claims), check numbers against union of cited quotes
        check_quotes = [q for q in (all_quotes or [e_clean]) if q.strip()]
        combined_nums = set()
        for q in check_quotes:
            combined_nums.update(extract_substantive_numbers(q))
        combined_quote_text = " ".join(check_quotes)

        c_nums = extract_substantive_numbers(c_stripped)
        for num in c_nums:
            if num not in combined_nums and num not in combined_quote_text:
                return NLIVerificationResult(
                    label=NLILabel.NOT_SUPPORTED,
                    confidence=0.90,
                    reason=f"Substantive metric number {num} in claim not found in cited evidence quote(s)",
                    verifier_type="rule_based",
                    claim_text=c_clean,
                    evidence_quote=e_clean
                )

        # 5. Token alignment (Token overlap alone NEVER returns SUPPORTED >= 0.85, only PARTIALLY_SUPPORTED)
        c_tokens = _tokenize_content_words(c_stripped)
        e_tokens = _tokenize_content_words(e_clean)

        if not c_tokens:
            return NLIVerificationResult(
                label=NLILabel.NOT_SUPPORTED,
                confidence=0.8,
                reason="No substantive content tokens in claim",
                verifier_type="rule_based",
                claim_text=c_clean,
                evidence_quote=e_clean
            )

        e_token_set = set(e_tokens)
        c_token_set = set(c_tokens)

        overlap = sum(1 for token in c_token_set if token in e_token_set or any(token in e_tok or e_tok in token for e_tok in e_token_set))
        overlap_ratio = overlap / max(1, len(c_token_set))

        if overlap_ratio >= 0.65:
            return NLIVerificationResult(
                label=NLILabel.PARTIALLY_SUPPORTED,
                confidence=0.70,
                reason=f"Strong token alignment ({overlap_ratio:.0%}) without exact match, deferring to LLM for full entailment",
                verifier_type="rule_based",
                claim_text=c_clean,
                evidence_quote=e_clean,
                metadata={"overlap_ratio": overlap_ratio}
            )
        elif overlap_ratio >= 0.40:
            return NLIVerificationResult(
                label=NLILabel.PARTIALLY_SUPPORTED,
                confidence=0.60,
                reason=f"Partial semantic token alignment ({overlap_ratio:.0%})",
                verifier_type="rule_based",
                claim_text=c_clean,
                evidence_quote=e_clean,
                metadata={"overlap_ratio": overlap_ratio}
            )
        else:
            return NLIVerificationResult(
                label=NLILabel.NOT_SUPPORTED,
                confidence=0.85,
                reason=f"Insufficient token alignment ({overlap_ratio:.0%}) with cited evidence quote",
                verifier_type="rule_based",
                claim_text=c_clean,
                evidence_quote=e_clean,
                metadata={"overlap_ratio": overlap_ratio}
            )


class LLMNLIVerifier(BaseNLIVerifier):
    """
    LLM-based Entailment Verifier with budget enforcement (Plan 23 Layer 3).
    """

    def __init__(self, llm: LLMBackend, fallback: Optional[BaseNLIVerifier] = None, budget_tracker: Optional[Any] = None):
        self.llm = llm
        self.fallback = fallback or RuleBasedNLIVerifier()
        self.budget_tracker = budget_tracker

    def verify(
        self,
        claim: str,
        evidence: str,
        budget_tracker: Optional[Any] = None,
        all_quotes: Optional[List[str]] = None,
        known_entities: Optional[List[str]] = None
    ) -> NLIVerificationResult:
        tracker = budget_tracker or self.budget_tracker
        if tracker:
            try:
                tracker.assert_can_call_llm()
            except Exception as e:
                logger.warning(f"LLM NLI skipped due to budget/time constraint ({e}). Falling back to rule-based.")
                return self.fallback.verify(
                    claim, evidence, budget_tracker=tracker, all_quotes=all_quotes, known_entities=known_entities
                )

        claim_to_check = clean_claim_for_nli(claim) or claim
        prompt = (
            f"You are a strict factual entailment checker.\n"
            f"Determine whether the Evidence text entails, contradicts, partially supports, or does not support the Claim.\n\n"
            f"Evidence: \"{evidence}\"\n"
            f"Claim: \"{claim_to_check}\"\n\n"
            f"Classification rules:\n"
            f"- SUPPORTED: Evidence explicitly confirms and proves the claim with matching model, metric, and numbers.\n"
            f"- PARTIALLY_SUPPORTED: Evidence provides some facts but leaves key parts unproven or ambiguous.\n"
            f"- NOT_SUPPORTED: Evidence does not contain relevant information to judge the claim, OR mentions a DIFFERENT model/entity, DIFFERENT dataset/benchmark, or DIFFERENT variant (e.g. evidence about Model A does NOT contradict a claim about Model B; attributing numbers to another model or dataset is NOT_SUPPORTED, not CONTRADICTED).\n"
            f"- CONTRADICTED: Evidence directly discusses the EXACT SAME model/entity and explicitly refutes the claim (e.g. reversed comparison direction, opposite polarity, or conflicting numbers for that exact same entity)."
        )

        try:
            res = self.llm.structured_generate(prompt, schema=NLIStructuredOutputSchema, num_predict=128)
            if tracker:
                calls = getattr(res, "calls_made", 1)
                tokens = getattr(res, "total_tokens", 0)
                if not isinstance(calls, int):
                    calls = 1
                if not isinstance(tokens, int):
                    tokens = 0
                tracker.record_llm_call(tokens=tokens, count=calls)

            parsed: NLIStructuredOutputSchema = res.parsed
            label = parsed.label
            confidence = parsed.confidence
            reason = parsed.reason

            # Guardrail: Distinguish genuine CONTRADICTED from attribution / entity / dataset mismatch
            if label == NLILabel.CONTRADICTED:
                is_genuine, conflict_reason = self._is_genuine_contradiction(
                    claim, evidence, known_entities=known_entities
                )
                if not is_genuine:
                    logger.info(f"[NLI Guardrail] Downgrading LLM CONTRADICTED to NOT_SUPPORTED ({conflict_reason}) for claim: '{claim[:60]}...'")
                    label = NLILabel.NOT_SUPPORTED
                    confidence = 0.85
                    reason = f"Entity/dataset mismatch: {conflict_reason}"

            return NLIVerificationResult(
                label=label,
                confidence=confidence,
                reason=reason,
                verifier_type="llm",
                claim_text=claim,
                evidence_quote=evidence,
                prompt_version=NLI_PROMPT_VERSION
            )
        except Exception as e:
            logger.warning(f"LLM NLI verification failed ({e}). Falling back to rule-based verifier.")
            return self.fallback.verify(
                claim, evidence, budget_tracker=tracker, all_quotes=all_quotes, known_entities=known_entities
            )

    def _is_genuine_contradiction(
        self,
        claim: str,
        evidence: str,
        known_entities: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        Validates whether an LLM CONTRADICTED verdict represents a genuine semantic/factual conflict
        or an entity/dataset mismatch that should be downgraded to NOT_SUPPORTED.
        """
        c_clean = strip_citation_markers(claim.strip())
        e_clean = evidence.strip()

        # 1. Rule verifier detected contradiction (reversed comparison, opposite polarity, etc.)
        rule_res = self.fallback.verify(c_clean, e_clean, known_entities=known_entities)
        if rule_res.label == NLILabel.CONTRADICTED:
            return True, rule_res.reason

        # 2. Verbal negation mismatch with matching core entities (e.g. pair #14)
        c_has_neg = any(re.search(rf"\b{re.escape(neg)}\b", c_clean.lower()) for neg in VERBAL_NEGATIONS)
        e_has_neg = any(re.search(rf"\b{re.escape(neg)}\b", e_clean.lower()) for neg in VERBAL_NEGATIONS)
        if c_has_neg != e_has_neg:
            c_ents = extract_core_entities(c_clean)
            e_ents = extract_core_entities(e_clean)
            if any(ce in e_ents or any(_entities_match(ce, ee) for ee in e_ents) for ce in c_ents):
                return True, "Direct verbal negation conflict on matching entities"

        # 3. Check for Entity / Dataset / Variant mismatch:
        # If claim asserts facts about an entity, dataset, or variant that is absent from evidence,
        # the evidence cannot contradict it (e.g. quote about YOLOv10 cannot contradict a claim about YOLOv8).
        c_ents = [
            e for e in extract_core_entities(c_clean)
            if e not in CONTEXT_EXCLUDED and not _is_number_or_unit(e)
        ]
        e_ents = [
            e for e in extract_core_entities(e_clean)
            if e not in CONTEXT_EXCLUDED and not _is_number_or_unit(e)
        ]

        if c_ents and e_ents:
            shared_entity = any(
                any(_entities_match(ce, ee) for ee in e_ents)
                for ce in c_ents
            )
            if not shared_entity:
                return False, f"Claim entities {c_ents} not discussed in evidence entities {e_ents}"

        c_datasets = {e for e in extract_core_entities(c_clean) if e in DATASET_BENCHMARK_TERMS}
        e_datasets = {e for e in extract_core_entities(e_clean) if e in DATASET_BENCHMARK_TERMS}
        if c_datasets and e_datasets and not (c_datasets & e_datasets):
            return False, f"Claim references dataset {c_datasets} but evidence references {e_datasets}"

        # 4. Check for conflicting numbers on the same entity and metric (e.g. pair #25)
        c_nums = extract_substantive_numbers(c_clean)
        e_nums = extract_substantive_numbers(e_clean)
        if c_nums and e_nums:
            c_subj = c_ents[0] if c_ents else None
            if c_subj and any(_entities_match(c_subj, ee) for ee in e_ents):
                if not set(c_nums).issubset(set(e_nums)):
                    return True, f"Conflicting numeric metrics for {c_subj} (claim has {c_nums}, evidence has {e_nums})"

        # 5. Check comparative triples
        c_triples = extract_comparative_triples(c_clean, known_entities=known_entities)
        e_triples = extract_comparative_triples(e_clean, known_entities=known_entities)
        if c_triples and (e_triples or e_ents):
            return True, "Comparative assertion evaluated against evidence"

        return False, "Evidence does not substantiate a direct contradiction against the claimed entity"


class CompositeNLIVerifier(BaseNLIVerifier):
    """
    Composite verifier combining fast CPU rule-based checks and LLM verification with budget recording.
    """

    def __init__(self, llm: Optional[LLMBackend] = None, budget_tracker: Optional[Any] = None):
        self.rule_verifier = RuleBasedNLIVerifier()
        self.llm_verifier = LLMNLIVerifier(llm=llm, fallback=self.rule_verifier, budget_tracker=budget_tracker) if llm else None

    def verify(
        self,
        claim: str,
        evidence: str,
        budget_tracker: Optional[Any] = None,
        all_quotes: Optional[List[str]] = None,
        known_entities: Optional[List[str]] = None
    ) -> NLIVerificationResult:
        # 1. Fast rule-based check
        rule_res = self.rule_verifier.verify(
            claim, evidence, budget_tracker=budget_tracker, all_quotes=all_quotes, known_entities=known_entities
        )

        # 2. If rule-based clearly identifies SUPPORTED or CONTRADICTED, return immediately (0 token cost!)
        if rule_res.label in (NLILabel.SUPPORTED, NLILabel.CONTRADICTED) and rule_res.confidence >= 0.85:
            return rule_res

        # 3. If rule-based is uncertain (PARTIALLY_SUPPORTED or NOT_SUPPORTED) and LLM is available, consult LLM
        if self.llm_verifier:
            return self.llm_verifier.verify(
                claim, evidence, budget_tracker=budget_tracker, all_quotes=all_quotes, known_entities=known_entities
            )

        return rule_res


class ClaimVerificationPipeline:
    """
    Manages multi-evidence entailment verification for claims across research sessions.
    """

    def __init__(self, verifier: Optional[BaseNLIVerifier] = None):
        self.verifier = verifier or RuleBasedNLIVerifier()
        self.rule_verifier = RuleBasedNLIVerifier()

    def verify_claim_against_quotes(
        self,
        claim_text: str,
        quotes: List[str],
        budget_tracker: Optional[Any] = None,
        known_entities: Optional[List[str]] = None
    ) -> NLIVerificationResult:
        """
        Evaluates a claim against multiple cited quotes.
        For multi-quote claims (Review item 2):
        1. Runs fast rule checks on individual quotes (0 LLM calls) to catch verbatim matches or direct contradictions.
        2. Calls verifier (at most 1 LLM call) on combined evidence ' '.join(valid_quotes) so the LLM sees complete context.
        """
        if not quotes:
            return NLIVerificationResult(
                label=NLILabel.NOT_SUPPORTED,
                confidence=1.0,
                reason="No evidence quotes provided for citation",
                verifier_type="pipeline",
                claim_text=claim_text,
                evidence_quote=""
            )

        valid_quotes = [q for q in quotes if q.strip()]
        if not valid_quotes:
            return NLIVerificationResult(
                label=NLILabel.NOT_SUPPORTED,
                confidence=1.0,
                reason="All provided evidence quotes were empty",
                verifier_type="pipeline",
                claim_text=claim_text,
                evidence_quote=""
            )

        if len(valid_quotes) == 1:
            return self.verifier.verify(
                claim_text,
                valid_quotes[0],
                budget_tracker=budget_tracker,
                all_quotes=valid_quotes,
                known_entities=known_entities
            )

        # Multi-quote claim optimization (Review item 2):
        # 1. Run zero-cost rule checks on individual quotes (0 LLM calls):
        # - Catches direct verbatim substring matches (SUPPORTED 1.0)
        # - Catches incontrovertible rule contradictions (reversed directions, numeric mismatch on single quote)
        # Note: We do NOT call LLM on individual quotes because:
        #   (a) It would consume N+1 LLM calls.
        #   (b) An individual quote in a multi-citation claim only has partial context (e.g. entity A),
        #       causing a 3B LLM to falsely emit CONTRADICTED or NOT_SUPPORTED.
        for q in valid_quotes:
            rule_res = self.rule_verifier.verify(
                claim_text,
                q,
                budget_tracker=budget_tracker,
                all_quotes=valid_quotes,
                known_entities=known_entities
            )
            # Incontrovertible contradiction on individual quote
            if rule_res.label == NLILabel.CONTRADICTED and rule_res.confidence >= 0.85:
                return rule_res
            # Direct exact match on individual quote
            if rule_res.label == NLILabel.SUPPORTED and rule_res.confidence >= 0.95:
                return rule_res

        # 2. Evaluate against the full combined evidence of all cited quotes with self.verifier (at most 1 LLM call!)
        combined_evidence = " ".join(valid_quotes)
        return self.verifier.verify(
            claim_text,
            combined_evidence,
            budget_tracker=budget_tracker,
            all_quotes=valid_quotes,
            known_entities=known_entities
        )
