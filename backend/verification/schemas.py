from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NLILabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class NLIVerificationResult(BaseModel):
    """Result of verifying entailment between a claim and evidence quote(s)."""
    label: NLILabel
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    reason: str = Field(default="", description="Explanation of entailment decision")
    verifier_type: str = Field(default="rule_based", description="Verifier implementation used")
    claim_text: str = Field(default="", description="The evaluated claim")
    evidence_quote: str = Field(default="", description="The cited evidence quote")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional debug/metric info")


class NLIStructuredOutputSchema(BaseModel):
    """Schema for LLM-based structured NLI responses."""
    label: NLILabel = Field(description="Entailment classification label")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    reason: str = Field(description="Concise rationale for why the evidence supports or contradicts the claim")
