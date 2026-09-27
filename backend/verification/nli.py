import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Set

from backend.verification.schemas import NLILabel, NLIVerificationResult, NLIStructuredOutputSchema
from backend.llm.backend import LLMBackend
from backend.core.coverage import (
    extract_substantive_numbers,
    subject_matches_entity,
    DATASET_BENCHMARK_TERMS,
)

logger = logging.getLogger(__name__)

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

CONTEXT_EXCLUDED = DATASET_BENCHMARK_TERMS | HARDWARE_ENV_TERMS | LINKING_VERBS

# Metric polarities (Plan 23 & Review items 3)
LOWER_IS_BETTER_METRICS: Set[str] = {
    "latency", "error", "error rate", "loss", "runtime", "time", "ms",
    "params", "parameters", "flops", "memory", "footprint", "vram", "cost", "delay"
}
HIGHER_IS_BETTER_METRICS: Set[str] = {
    "accuracy", "acc", "ap", "map", "nds", "throughput", "fps", "score",
    "f1", "precision", "recall", "speedup", "tps"
}
ALL_KNOWN_METRICS: Set[str] = LOWER_IS_BETTER_METRICS | HIGHER_IS_BETTER_METRICS

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


def strip_citation_markers(text: str) -> str:
    """Removes [E1], **E1**, (E1), E1 citation tags from sentence."""
    t = re.sub(r"\[E\d+\]", " ", text)
    t = re.sub(r"\*\*E\d+\*\*", " ", t)
    t = re.sub(r"\(E\d+\)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _normalize_text_for_match(text: str) -> str:
    """Keep decimals like 67.3 intact, strip punctuation and citation brackets, lowercase and whitespace-normalize."""
    t = strip_citation_markers(text).lower()
    t = re.sub(r"(?<=\d)\.(?=\d)", "__DOT__", t)
    t = re.sub(r"[^a-zA-Z0-9_\-\s]", " ", t)
    t = t.replace("__DOT__", ".")
    return re.sub(r"\s+", " ", t).strip()


def _entities_match(e1: str, e2: str) -> bool:
    """Checks whether two entities are identical or variants of the same model/subject."""
    if not e1 or not e2:
        return False
    e1_l, e2_l = e1.lower().strip(), e2.lower().strip()
    if e1_l == e2_l:
        return True
    return subject_matches_entity(e1_l, e2_l) or subject_matches_entity(e2_l, e1_l)


def _clean_token(t: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "", t.lower())


def _tokenize_content_words(text: str) -> List[str]:
    """Extract substantive lowercased alphanumeric content words excluding common stopwords."""
    words = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def extract_comparative_triples(text: str) -> List[Tuple[str, str, str]]:
    """
    Extracts structured comparative relations from text: (subject, comparative_word, object).
    Uses strict word boundaries and backward entity search for the subject.
    """
    clean_text = strip_citation_markers(text).lower()
    triples: List[Tuple[str, str, str]] = []

    # 1. Pattern: <EntityA> ... <comp_word> ... than <EntityB>
    than_matches = list(re.finditer(r"\bthan\s+([a-zA-Z0-9_\-\.]+)\b", clean_text))
    for tm in than_matches:
        obj_entity = tm.group(1).strip()
        before_than = clean_text[:tm.start()]

        # Search for comparative words using word boundaries
        found_comps: List[Tuple[int, int, str]] = []
        for comp in ALL_COMPARATIVES:
            for cm in re.finditer(rf"\b{re.escape(comp)}\b", before_than):
                found_comps.append((cm.start(), cm.end(), comp))

        found_comps.sort(key=lambda x: x[0])

        for c_start, c_end, comp in found_comps:
            text_before_comp = before_than[:c_start].strip()
            tokens_before = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text_before_comp)

            # Search backwards from comparative word to find closest substantive subject
            sub_entity = None
            for tok in reversed(tokens_before):
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
    verb_pattern = r"\b([a-zA-Z0-9_\-\.]+)\b\s+(?:[a-zA-Z0-9_\-\.]+\s+){0,2}?\b(outperforms|underperforms|beats|trails|exceeds)\b\s+\b([a-zA-Z0-9_\-\.]+)\b"
    for m in re.finditer(verb_pattern, clean_text):
        sub, comp, obj = m.group(1), m.group(2), m.group(3)
        if sub not in STOPWORDS and obj not in STOPWORDS and not _entities_match(sub, obj):
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
        all_quotes: Optional[List[str]] = None
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
        all_quotes: Optional[List[str]] = None
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
        c_triples = extract_comparative_triples(c_clean)
        e_triples = extract_comparative_triples(e_clean)

        for c_sub, c_comp, c_obj in c_triples:
            # Check if this triple is already explicitly supported by evidence
            has_agreeing = any(
                _entities_match(c_sub, e_s) and _entities_match(c_obj, e_o) and (
                    c_comp == e_c or (
                        c_comp in SUPERIOR_TERMS and e_c in SUPERIOR_TERMS
                    ) or (
                        c_comp in INFERIOR_TERMS and e_c in INFERIOR_TERMS
                    )
                )
                for e_s, e_c, e_o in e_triples
            )
            if has_agreeing:
                continue

            for e_sub, e_comp, e_obj in e_triples:
                c_metrics = _extract_metric_terms(c_clean)
                e_metrics = _extract_metric_terms(e_clean)
                shared_metrics = c_metrics & e_metrics

                # 3a. Reversed comparison direction (e.g. "A is faster than B" vs "B is faster than A")
                if _entities_match(c_sub, e_obj) and _entities_match(c_obj, e_sub):
                    if c_comp == e_comp:
                        # Exactly identical comparative word on swapped entities
                        # E.g. A faster than B vs B faster than A -> contradiction
                        if not (c_metrics and e_metrics and not shared_metrics):
                            return NLIVerificationResult(
                                label=NLILabel.CONTRADICTED,
                                confidence=0.95,
                                reason=f"Reversed comparison direction between {c_sub} and {c_obj}: claim states '{c_sub} {c_comp} {c_obj}', but evidence states '{e_sub} {e_comp} {e_obj}'",
                                verifier_type="rule_based",
                                claim_text=c_clean,
                                evidence_quote=e_clean
                            )
                    elif c_comp in SUPERIOR_TERMS and e_comp in SUPERIOR_TERMS:
                        if shared_metrics or not (c_metrics or e_metrics):
                            return NLIVerificationResult(
                                label=NLILabel.CONTRADICTED,
                                confidence=0.95,
                                reason=f"Reversed superior comparison between {c_sub} and {c_obj}: claim states '{c_sub} {c_comp} {c_obj}', but evidence states '{e_sub} {e_comp} {e_obj}'",
                                verifier_type="rule_based",
                                claim_text=c_clean,
                                evidence_quote=e_clean
                            )
                    elif c_comp in INFERIOR_TERMS and e_comp in INFERIOR_TERMS:
                        if shared_metrics or not (c_metrics or e_metrics):
                            return NLIVerificationResult(
                                label=NLILabel.CONTRADICTED,
                                confidence=0.95,
                                reason=f"Reversed inferior comparison between {c_sub} and {c_obj}: claim states '{c_sub} {c_comp} {c_obj}', but evidence states '{e_sub} {e_comp} {e_obj}'",
                                verifier_type="rule_based",
                                claim_text=c_clean,
                                evidence_quote=e_clean
                            )

                # 3b. Opposite comparative direction on same entity pair (e.g. "faster" vs "slower")
                if _entities_match(c_sub, e_sub) and _entities_match(c_obj, e_obj):
                    # Case I: Inherent metric-specific opposing pairs (faster vs slower, outperforms vs underperforms)
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
                    # E.g. "higher latency" vs "lower latency" -> CONTRADICTED
                    # "higher accuracy" vs "lower accuracy" -> CONTRADICTED
                    # But "lower error rate" vs "higher accuracy" -> NOT CONTRADICTED (different metric words, aligned polarity)
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
        all_quotes: Optional[List[str]] = None
    ) -> NLIVerificationResult:
        tracker = budget_tracker or self.budget_tracker
        if tracker:
            try:
                tracker.assert_can_call_llm()
            except Exception as e:
                logger.warning(f"LLM NLI skipped due to budget/time constraint ({e}). Falling back to rule-based.")
                return self.fallback.verify(claim, evidence, budget_tracker=tracker, all_quotes=all_quotes)

        prompt = (
            f"You are a strict factual entailment checker.\n"
            f"Determine whether the Evidence text entails, contradicts, partially supports, or does not support the Claim.\n\n"
            f"Evidence: \"{evidence}\"\n"
            f"Claim: \"{claim}\"\n\n"
            f"Classification rules:\n"
            f"- SUPPORTED: Evidence explicitly confirms and proves the claim.\n"
            f"- PARTIALLY_SUPPORTED: Evidence provides some facts but leaves key parts unproven or ambiguous.\n"
            f"- NOT_SUPPORTED: Evidence does not contain relevant information to judge the claim.\n"
            f"- CONTRADICTED: Evidence directly refutes or contradicts the claim."
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
            return NLIVerificationResult(
                label=parsed.label,
                confidence=parsed.confidence,
                reason=parsed.reason,
                verifier_type="llm",
                claim_text=claim,
                evidence_quote=evidence
            )
        except Exception as e:
            logger.warning(f"LLM NLI verification failed ({e}). Falling back to rule-based verifier.")
            return self.fallback.verify(claim, evidence, budget_tracker=tracker, all_quotes=all_quotes)


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
        all_quotes: Optional[List[str]] = None
    ) -> NLIVerificationResult:
        # 1. Fast rule-based check
        rule_res = self.rule_verifier.verify(claim, evidence, budget_tracker=budget_tracker, all_quotes=all_quotes)

        # 2. If rule-based clearly identifies SUPPORTED or CONTRADICTED, return immediately (0 token cost!)
        if rule_res.label in (NLILabel.SUPPORTED, NLILabel.CONTRADICTED) and rule_res.confidence >= 0.85:
            return rule_res

        # 3. If rule-based is uncertain (PARTIALLY_SUPPORTED or NOT_SUPPORTED) and LLM is available, consult LLM
        if self.llm_verifier:
            return self.llm_verifier.verify(claim, evidence, budget_tracker=budget_tracker, all_quotes=all_quotes)

        return rule_res


class ClaimVerificationPipeline:
    """
    Manages multi-evidence entailment verification for claims across research sessions.
    """

    def __init__(self, verifier: Optional[BaseNLIVerifier] = None):
        self.verifier = verifier or RuleBasedNLIVerifier()

    def verify_claim_against_quotes(self, claim_text: str, quotes: List[str], budget_tracker: Optional[Any] = None) -> NLIVerificationResult:
        """
        Evaluates a claim against multiple cited quotes.
        Priority:
        1. CONTRADICTED: if any cited quote directly refutes the claim.
        2. SUPPORTED: if any cited quote explicitly supports the claim, or if combined quotes support it.
        3. PARTIALLY_SUPPORTED: if partial overlap is found.
        4. NOT_SUPPORTED: otherwise.
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

        # 1. Check each quote individually for direct CONTRADICTION or single-quote SUPPORT
        results: List[NLIVerificationResult] = []
        for q in valid_quotes:
            res = self.verifier.verify(claim_text, q, budget_tracker=budget_tracker, all_quotes=valid_quotes)
            results.append(res)
            # Short-circuit on direct contradiction!
            if res.label == NLILabel.CONTRADICTED:
                return res

        # 2. Check for single-quote explicit support
        for res in results:
            if res.label == NLILabel.SUPPORTED:
                return res

        # 3. For multi-quote claims, evaluate against the combined evidence of all cited quotes!
        if len(valid_quotes) > 1:
            combined_evidence = " ".join(valid_quotes)
            comb_res = self.verifier.verify(
                claim_text, combined_evidence, budget_tracker=budget_tracker, all_quotes=valid_quotes
            )
            if comb_res.label == NLILabel.CONTRADICTED:
                return comb_res
            if comb_res.label == NLILabel.SUPPORTED:
                return comb_res
            results.append(comb_res)

        # 4. Check for partial support
        for res in results:
            if res.label == NLILabel.PARTIALLY_SUPPORTED:
                return res

        return results[0]
