import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Tokenizer preserving metrics (e.g. 71.2), version tags (e.g. v2.1, flashattention-2), and technical names
TOKEN_REGEX = re.compile(r"[a-zA-Z0-9]+(?:[\.\-_][a-zA-Z0-9]+)*")


def tokenize_for_bm25(text: str) -> List[str]:
    """Extracts normalized lowercase tokens while preserving metric values and hyphenated identifiers."""
    if not text:
        return []
    return [t.lower() for t in TOKEN_REGEX.findall(text)]


class BM25Index:
    """
    In-memory BM25 index powered by rank_bm25.BM25Okapi.
    Specialized for keyword, metric, model name, and exact-value queries.
    """

    def __init__(self, chunks: Optional[List[Dict[str, Any]]] = None):
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        if chunks:
            self.index(chunks)

    def index(self, chunks: List[Dict[str, Any]]) -> None:
        """Indexes a list of chunk dictionaries. Each chunk must contain a 'text' field."""
        self.chunks = chunks
        if not chunks:
            self.bm25 = None
            return

        corpus = [tokenize_for_bm25(c.get("text", "")) for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"[BM25Index] Indexed {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches the BM25 index with a text query.
        Returns top_k tuples of (chunk_dict, score) sorted descending by score.
        """
        if not self.bm25 or not self.chunks:
            return []

        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        scored_indices = sorted(
            [(i, float(scores[i])) for i in range(len(scores))],
            key=lambda x: x[1],
            reverse=True
        )

        results: List[Tuple[Dict[str, Any], float]] = []
        for idx, score in scored_indices[:top_k]:
            if score > 0.0:  # Only return chunks with at least minimal keyword overlap
                results.append((self.chunks[idx], score))

        return results
