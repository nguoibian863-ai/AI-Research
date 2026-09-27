import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Set

from backend.verification.schemas import NLILabel, NLIVerificationResult, NLIStructuredOutputSchema
from backend.llm.backend import LLMBackend

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could",
    "will", "would", "shall", "should", "it", "its", "that", "this", "these", "those",
    "which", "who", "whom", "what", "as", "if", "than", "so", "than", "too", "very"
}

ANTONYM_PAIRS: List[Tuple[str, str]] = [
    ("faster", "slower"),
    ("higher", "lower"),
    ("more", "less"),
    ("better", "worse"),
    ("superior", "inferior"),
    ("outperforms", "underperforms"),
    ("exceeds", "trails"),
    ("increase", "decrease"),
    ("improves", "degrades"),
    ("increases", "decreases"),
    ("improved", "degraded"),
    ("exceeded", "trailed")
]

NEGATION_TERMS: Set[str] = {
    "not", "no", "never", "cannot", "cant", "fails", "fail", "failed", "failing",
    "unable", "without", "neither", "nor"
}


def _tokenize_content_words(text: str) -> List[str]:
    """Extract lowercased alphanumeric content words excluding common stopwords."""
    words = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


class BaseNLIVerifier(ABC):
    """Abstract base class for Natural Language Inference / Entailment Verifiers (Plan 23)."""

    @abstractmethod
    def verify(self, claim: str, evidence: str) -> NLIVerificationResult:
        """Verify entailment between claim and evidence."""
        pass

    def verify_batch(self, pairs: List[Tuple[str, str]]) -> List[NLIVerificationResult]:
        """Verify multiple claim-evidence pairs."""
        return [self.verify(claim, evidence) for claim, evidence in pairs]


class RuleBasedNLIVerifier(BaseNLIVerifier):
    """
    High-precision, zero-latency rule-based entailment checker.
    Evaluates token overlap, comparative direction consistency, and negation polarity.
    """

    def verify(self, claim: str, evidence: str) -> NLIVerificationResult:
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

        # 1. Check for comparative/directional contradiction
        for word_a, word_b in ANTONYM_PAIRS:
            if word_a in c_token_set and word_b in e_token_set:
                return NLIVerificationResult(
                    label=NLILabel.CONTRADICTED,
                    confidence=0.9,
                    reason=f"Directional contradiction: claim states '{word_a}' but evidence states '{word_b}'",
                    verifier_type="rule_based",
                    claim_text=c_clean,
                    evidence_quote=e_clean
                )
            if word_b in c_token_set and word_a in e_token_set:
                return NLIVerificationResult(
                    label=NLILabel.CONTRADICTED,
                    confidence=0.9,
                    reason=f"Directional contradiction: claim states '{word_b}' but evidence states '{word_a}'",
                    verifier_type="rule_based",
                    claim_text=c_clean,
                    evidence_quote=e_clean
                )

        # 2. Check for negation polarity mismatch
        c_has_neg = any(neg in c_token_set for neg in NEGATION_TERMS)
        e_has_neg = any(neg in e_token_set for neg in NEGATION_TERMS)

        overlap = sum(1 for token in c_token_set if token in e_token_set or any(token in e_tok or e_tok in token for e_tok in e_token_set))
        overlap_ratio = overlap / max(1, len(c_token_set))

        if c_has_neg != e_has_neg and overlap_ratio >= 0.5:
            return NLIVerificationResult(
                label=NLILabel.CONTRADICTED,
                confidence=0.85,
                reason="Negation polarity mismatch between claim and evidence",
                verifier_type="rule_based",
                claim_text=c_clean,
                evidence_quote=e_clean,
                metadata={"overlap_ratio": overlap_ratio}
            )

        # 3. Overlap thresholding
        if overlap_ratio >= 0.60:
            return NLIVerificationResult(
                label=NLILabel.SUPPORTED,
                confidence=min(1.0, 0.75 + (overlap_ratio * 0.25)),
                reason=f"Strong semantic token alignment ({overlap_ratio:.0%}) with consistent direction",
                verifier_type="rule_based",
                claim_text=c_clean,
                evidence_quote=e_clean,
                metadata={"overlap_ratio": overlap_ratio}
            )
        elif overlap_ratio >= 0.35:
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
                reason=f"Insufficient token alignment ({overlap_ratio:.0%}) with evidence",
                verifier_type="rule_based",
                claim_text=c_clean,
                evidence_quote=e_clean,
                metadata={"overlap_ratio": overlap_ratio}
            )


class LLMNLIVerifier(BaseNLIVerifier):
    """
    LLM-based Entailment Verifier using structured generation (Plan 23 Layer 3).
    """

    def __init__(self, llm: LLMBackend, fallback: Optional[BaseNLIVerifier] = None):
        self.llm = llm
        self.fallback = fallback or RuleBasedNLIVerifier()

    def verify(self, claim: str, evidence: str) -> NLIVerificationResult:
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
            return self.fallback.verify(claim, evidence)


class CompositeNLIVerifier(BaseNLIVerifier):
    """
    Composite verifier combining LLM and Rule-based verifiers with automatic fallback.
    """

    def __init__(self, llm: Optional[LLMBackend] = None):
        self.rule_verifier = RuleBasedNLIVerifier()
        self.llm_verifier = LLMNLIVerifier(llm=llm, fallback=self.rule_verifier) if llm else None

    def verify(self, claim: str, evidence: str) -> NLIVerificationResult:
        if self.llm_verifier:
            return self.llm_verifier.verify(claim, evidence)
        return self.rule_verifier.verify(claim, evidence)


class ClaimVerificationPipeline:
    """
    Manages multi-evidence entailment verification for claims across research sessions.
    """

    def __init__(self, verifier: Optional[BaseNLIVerifier] = None):
        self.verifier = verifier or RuleBasedNLIVerifier()

    def verify_claim_against_quotes(self, claim_text: str, quotes: List[str]) -> NLIVerificationResult:
        """
        Evaluates a claim against multiple cited quotes.
        If any quote CONTRADICTS -> returns CONTRADICTED.
        If any quote is SUPPORTED -> returns SUPPORTED.
        Otherwise returns highest confidence partial/not supported result.
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

        results = [self.verifier.verify(claim_text, q) for q in quotes if q.strip()]
        if not results:
            return NLIVerificationResult(
                label=NLILabel.NOT_SUPPORTED,
                confidence=1.0,
                reason="All provided evidence quotes were empty",
                verifier_type="pipeline",
                claim_text=claim_text,
                evidence_quote=""
            )

        # 1. Contradiction takes precedence
        for res in results:
            if res.label == NLILabel.CONTRADICTED:
                return res

        # 2. Supported takes precedence over partial
        for res in results:
            if res.label == NLILabel.SUPPORTED:
                return res

        # 3. Partially supported takes precedence over not supported
        for res in results:
            if res.label == NLILabel.PARTIALLY_SUPPORTED:
                return res

        return results[0]
