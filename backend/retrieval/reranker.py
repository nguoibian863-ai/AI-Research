import re
import logging
from typing import List
from backend.retrieval.hybrid import RetrievedChunk

logger = logging.getLogger(__name__)


class ScoreReranker:
    """
    Reranks retrieved candidate chunks using multi-signal scoring:
    1. Exact query phrase and keyword overlap boost.
    2. Section title relevance boost (e.g. 'Results', 'Experiments', 'Benchmark' for quantitative queries).
    3. Metric / numerical match boost for benchmark and performance queries.
    """

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
        """Reranks candidates and returns the top_k sorted by boosted score descending."""
        if not chunks:
            return []

        query_lower = query.lower().strip()
        query_words = set(re.findall(r"\w+", query_lower))
        is_benchmark_query = bool(query_words & cls.BENCHMARK_KEYWORDS)

        reranked: List[RetrievedChunk] = []
        for c in chunks:
            text_lower = c.text.lower()
            section_lower = (c.section or "").lower()
            boost = 0.0

            # 1. Exact query match boost
            if query_lower in text_lower:
                boost += 0.25
            else:
                text_words = set(re.findall(r"\w+", text_lower))
                if query_words:
                    overlap_ratio = len(query_words & text_words) / len(query_words)
                    boost += 0.15 * overlap_ratio

            # 2. Section relevance boost
            if is_benchmark_query and any(sec in section_lower for sec in cls.BENCHMARK_SECTIONS):
                boost += 0.15

            # 3. Numeric metrics boost if query seeks numbers
            has_digits_in_query = any(char.isdigit() for char in query)
            has_digits_in_text = any(char.isdigit() for char in c.text)
            if has_digits_in_query and has_digits_in_text:
                boost += 0.10

            new_score = c.score + boost
            chunk_copy = c.model_copy(update={"score": new_score})
            reranked.append(chunk_copy)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
