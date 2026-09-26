"""Evidence Engine package: Atomic Evidence Extraction, Quote Invariance Verification, and Citation Verification."""
from backend.evidence.extractor import EvidenceExtractor, find_quote_in_text
from backend.evidence.verifier import CitationVerifier

__all__ = ["EvidenceExtractor", "find_quote_in_text", "CitationVerifier"]
