import re
import logging
from typing import List
from backend.retrieval.hybrid import RetrievedChunk

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "how", "what", "why", "where", "when",
    "does", "do", "did", "and", "or", "we", "you", "i", "it", "this", "that"
}


class ScoreReranker:
    """
    Reranks retrieved candidate chunks using calibrated multi-signal scoring without overpowering RRF:
    1. Exact query phrase match: applies when the full query or a key multi-word phrase appears verbatim.
    2. Benchmark/results section relevance: applies when query is benchmark/empirical and chunk is in Results/Experiments.
    3. Target numeric match: applies when query asks for a specific numeric target (e.g. '71.2 NDS') and the number matches.

    Crucially, boosts are scaled by 1.0 / (rrf_k + 1) (~0.016) so that reranking acts as a fine-grained
    tie-breaker / rank refinement rather than obliterating dense vector semantic search signals.
    """

    RRF_K = 60
    SCALE = 1.0 / (RRF_K + 1)  # ~0.01639

    BENCHMARK_KEYWORDS = {
        "benchmark", "result", "results", "experiment", "experiments",
        "evaluation", "accuracy", "latency", "fps", "score", "map", "nds",
        "speed", "runtime", "throughput", "vram", "memory"
    }

    BENCHMARK_SECTIONS = {
        "results", "experiments", "evaluation", "benchmark", "ablation", "performance"
    }

    @classmethod
    def rerank(cls, query: str, chunks: List[RetrievedChunk], top_k: int = 5) -> List[RetrievedChunk]:
        """Reranks candidates and returns the top_k sorted by calibrated score descending."""
        if not chunks:
            return []

        query_lower = query.lower().strip()
        query_words = set(re.findall(r"\w+", query_lower))
        content_words = query_words - STOP_WORDS
        is_benchmark_query = bool(content_words & cls.BENCHMARK_KEYWORDS)

        # Extract explicit target numbers from query (e.g. '71.2', '59.2', '3050')
        query_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", query)

        reranked: List[RetrievedChunk] = []
        for c in chunks:
            text_lower = c.text.lower()
            section_lower = (c.section or "").lower()
            raw_boost = 0.0

            # 1. Exact multi-word query phrase match (verbatim match)
            if len(query_lower) >= 4 and query_lower in text_lower:
                raw_boost += 0.20
            elif content_words:
                # Modest boost only if ALL key content words match
                text_words = set(re.findall(r"\w+", text_lower))
                if content_words.issubset(text_words):
                    raw_boost += 0.10

            # 2. Section relevance boost for empirical / benchmark queries
            if is_benchmark_query and any(sec in section_lower for sec in cls.BENCHMARK_SECTIONS):
                raw_boost += 0.10

            # 3. Explicit numeric target match (e.g., query explicitly contains '71.2' and chunk has '71.2')
            if query_numbers and any(num in text_lower for num in query_numbers):
                raw_boost += 0.10

            # Scale boost so it does not overpower RRF score
            scaled_boost = raw_boost * cls.SCALE
            new_score = c.score + scaled_boost
            chunk_copy = c.model_copy(update={"score": new_score})
            reranked.append(chunk_copy)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]

