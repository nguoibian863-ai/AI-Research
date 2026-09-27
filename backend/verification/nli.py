import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Set

from backend.verification.schemas import NLILabel, NLIVerificationResult, NLIStructuredOutputSchema
from backend.llm.backend import LLMBackend
from backend.core.coverage import extract_substantive_numbers

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could",
    "will", "would", "shall", "should", "it", "its", "that", "this", "these", "those",
    "which", "who", "whom", "what", "as", "if", "so", "too", "very", "our", "their", "of"
}

# Comparative directional categories
SUPERIOR_TERMS: Dict[str, str] = {
    "faster": "slower",
    "higher": "lower",
    "more": "less",
    "better": "worse",
    "superior": "inferior",
    "outperforms": "underperforms",
    "exceeds": "trails",
    "beats": "loses to",
    "greater": "smaller",
    "larger": "smaller"
}
INFERIOR_TERMS: Dict[str, str] = {v: k for k, v in SUPERIOR_TERMS.items()}
ALL_COMPARATIVES: Set[str] = set(SUPERIOR_TERMS.keys()) | set(INFERIOR_TERMS.keys())

# Explicit verbal negation markers (prepositions like "without" are intentionally excluded)
VERBAL_NEGATIONS: Set[str] = {
    "not", "never", "cannot", "cant", "fails to", "unable to", "neither", "nor"
}


def _clean_token(t: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "", t.lower())


def _tokenize_content_words(text: str) -> List[str]:
    """Extract substantive lowercased alphanumeric content words excluding common stopwords."""
    words = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def extract_comparative_triples(text: str) -> List[Tuple[str, str, str]]:
    """
    Extracts structured comparative relations from text: (subject, comparative_word, object).
    Examples:
      - 'YOLOv8 is faster than RT-DETR' -> [('yolov8', 'faster', 'rt-detr')]
      - 'RT-DETR achieves higher AP with lower latency than YOLOv8' ->
          [('rt-detr', 'higher', 'yolov8'), ('rt-detr', 'lower', 'yolov8')]
      - 'RT-DETR outperforms YOLOv8' -> [('rt-detr', 'outperforms', 'yolov8')]
    """
    t_lower = text.lower()
    triples: List[Tuple[str, str, str]] = []

    # 1. Pattern: <EntityA> ... <comp_word> (optional metric) than <EntityB>
    # Look for "than <EntityB>"
    than_matches = list(re.finditer(r"\bthan\s+([a-zA-Z0-9_\-\.]+)\b", t_lower))
    for tm in than_matches:
        obj_entity = tm.group(1)
        before_than = t_lower[:tm.start()]

        # Find any comparative words before "than"
        for comp in ALL_COMPARATIVES:
            comp_idx = before_than.rfind(comp)
            if comp_idx != -1:
                # Subject is the principal entity before the comparative
                text_before_comp = before_than[:comp_idx].strip()
                tokens_before = [w for w in re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text_before_comp) if w not in STOPWORDS]
                if tokens_before:
                    sub_entity = tokens_before[0]
                    if sub_entity != obj_entity:
                        triples.append((sub_entity, comp, obj_entity))

    # 2. Pattern: <EntityA> (outperforms|underperforms|beats|trails|exceeds) <EntityB>
    verb_pattern = r"\b([a-zA-Z0-9_\-\.]+)\b\s+(?:[a-zA-Z0-9_\-\.]+\s+){0,2}?\b(outperforms|underperforms|beats|trails|exceeds)\b\s+\b([a-zA-Z0-9_\-\.]+)\b"
    for m in re.finditer(verb_pattern, t_lower):
        sub, comp, obj = m.group(1), m.group(2), m.group(3)
        if sub not in STOPWORDS and obj not in STOPWORDS and sub != obj:
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
    def verify(self, claim: str, evidence: str, budget_tracker: Optional[Any] = None) -> NLIVerificationResult:
        """Verify entailment between claim and evidence."""
        pass

    def verify_batch(self, pairs: List[Tuple[str, str]], budget_tracker: Optional[Any] = None) -> List[NLIVerificationResult]:
        """Verify multiple claim-evidence pairs."""
        return [self.verify(claim, evidence, budget_tracker=budget_tracker) for claim, evidence in pairs]


class RuleBasedNLIVerifier(BaseNLIVerifier):
    """
    High-precision, zero-latency rule-based entailment checker.
    Specifically checks:
    1. Near-identity / substring match
    2. Entity-aware comparative relation and direction consistency
    3. Token overlap and numbers alignment
    """

    def verify(self, claim: str, evidence: str, budget_tracker: Optional[Any] = None) -> NLIVerificationResult:
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

        c_norm = re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9_\-\.\s]", "", c_clean.lower()))
        e_norm = re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9_\-\.\s]", "", e_clean.lower()))

        # 1. Exact or near-exact match
        if c_norm == e_norm or c_norm in e_norm or e_norm in c_norm:
            return NLIVerificationResult(
                label=NLILabel.SUPPORTED,
                confidence=1.0,
                reason="Claim is an exact or direct substring match of evidence quote",
                verifier_type="rule_based",
                claim_text=c_clean,
                evidence_quote=e_clean
            )

        # 2. Entity-aware comparative relation check (Resolves Review Item 2)
        c_triples = extract_comparative_triples(c_clean)
        e_triples = extract_comparative_triples(e_clean)

        for c_sub, c_comp, c_obj in c_triples:
            c_dir = _get_comp_direction(c_comp)
            for e_sub, e_comp, e_obj in e_triples:
                e_dir = _get_comp_direction(e_comp)
                same_entities = (c_sub == e_sub and c_obj == e_obj)
                swapped_entities = (c_sub == e_obj and c_obj == e_sub)

                if same_entities:
                    # Same entity order: direction must agree
                    if c_dir != 0 and e_dir != 0 and c_dir != e_dir:
                        return NLIVerificationResult(
                            label=NLILabel.CONTRADICTED,
                            confidence=0.95,
                            reason=f"Opposite comparative direction for ({c_sub} vs {c_obj}): claim states '{c_comp}' but evidence states '{e_comp}'",
                            verifier_type="rule_based",
                            claim_text=c_clean,
                            evidence_quote=e_clean
                        )
                elif swapped_entities:
                    # Swapped entities: e.g. "A is faster than B" vs "B is faster than A"
                    if c_dir != 0 and e_dir != 0 and c_dir == e_dir:
                        return NLIVerificationResult(
                            label=NLILabel.CONTRADICTED,
                            confidence=0.95,
                            reason=f"Reversed comparison direction between {c_sub} and {c_obj}: claim claims {c_sub} {c_comp} {c_obj}, but evidence states {e_sub} {e_comp} {e_obj}",
                            verifier_type="rule_based",
                            claim_text=c_clean,
                            evidence_quote=e_clean
                        )

        # 3. Numeric consistency check
        c_nums = extract_substantive_numbers(c_clean)
        e_nums = extract_substantive_numbers(e_clean)
        for num in c_nums:
            if num not in e_nums and num not in e_clean:
                return NLIVerificationResult(
                    label=NLILabel.NOT_SUPPORTED,
                    confidence=0.90,
                    reason=f"Substantive metric number {num} in claim not found in cited evidence quote",
                    verifier_type="rule_based",
                    claim_text=c_clean,
                    evidence_quote=e_clean
                )

        # 4. Token alignment
        c_tokens = _tokenize_content_words(c_clean)
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
                label=NLILabel.SUPPORTED,
                confidence=min(1.0, 0.80 + (overlap_ratio * 0.20)),
                reason=f"Strong semantic token alignment ({overlap_ratio:.0%}) with consistent facts",
                verifier_type="rule_based",
                claim_text=c_clean,
                evidence_quote=e_clean,
                metadata={"overlap_ratio": overlap_ratio}
            )
        elif overlap_ratio >= 0.40:
            return NLIVerificationResult(
                label=NLILabel.PARTIALLY_SUPPORTED,
                confidence=0.70,
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

    def verify(self, claim: str, evidence: str, budget_tracker: Optional[Any] = None) -> NLIVerificationResult:
        tracker = budget_tracker or self.budget_tracker
        if tracker:
            try:
                tracker.assert_can_call_llm()
            except Exception as e:
                logger.warning(f"LLM NLI skipped due to budget/time constraint ({e}). Falling back to rule-based.")
                return self.fallback.verify(claim, evidence, budget_tracker=tracker)

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
            return self.fallback.verify(claim, evidence, budget_tracker=tracker)


class CompositeNLIVerifier(BaseNLIVerifier):
    """
    Composite verifier combining fast CPU rule-based checks and LLM verification with budget recording.
    """

    def __init__(self, llm: Optional[LLMBackend] = None, budget_tracker: Optional[Any] = None):
        self.rule_verifier = RuleBasedNLIVerifier()
        self.llm_verifier = LLMNLIVerifier(llm=llm, fallback=self.rule_verifier, budget_tracker=budget_tracker) if llm else None

    def verify(self, claim: str, evidence: str, budget_tracker: Optional[Any] = None) -> NLIVerificationResult:
        # 1. Fast rule-based check
        rule_res = self.rule_verifier.verify(claim, evidence, budget_tracker=budget_tracker)

        # 2. If rule-based clearly identifies SUPPORTED or CONTRADICTED, return immediately (0 token cost!)
        if rule_res.label in (NLILabel.SUPPORTED, NLILabel.CONTRADICTED) and rule_res.confidence >= 0.85:
            return rule_res

        # 3. If rule-based is uncertain (PARTIALLY_SUPPORTED or NOT_SUPPORTED) and LLM is available, consult LLM
        if self.llm_verifier:
            return self.llm_verifier.verify(claim, evidence, budget_tracker=budget_tracker)

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
        2. SUPPORTED: if any cited quote explicitly supports the claim.
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

        results: List[NLIVerificationResult] = []
        for q in valid_quotes:
            res = self.verifier.verify(claim_text, q, budget_tracker=budget_tracker)
            results.append(res)
            # Short-circuit on direct contradiction!
            if res.label == NLILabel.CONTRADICTED:
                return res

        # Check for support
        for res in results:
            if res.label == NLILabel.SUPPORTED:
                return res

        for res in results:
            if res.label == NLILabel.PARTIALLY_SUPPORTED:
                return res

        return results[0]
