import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.retrieval.bm25 import BM25Index
from backend.retrieval.faiss_index import FaissVectorIndex

logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    char_start: int = 0
    char_end: int = 0
    score: float = 0.0
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HybridRetriever:
    """
    Hybrid Retriever merging keyword search (BM25) and semantic vector search (FAISS)
    via Reciprocal Rank Fusion (RRF).
    Fulfills Week 2 requirement: Hybrid retrieval combining BM25 exact match and vector semantic search.
    """

    def __init__(
        self,
        bm25_index: Optional[BM25Index] = None,
        vector_index: Optional[FaissVectorIndex] = None,
        rrf_k: int = 60,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5
    ):
        self.bm25_index = bm25_index or BM25Index()
        self.vector_index = vector_index or FaissVectorIndex()
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Indexes chunks in both BM25 and FAISS indices simultaneously."""
        self.bm25_index.index(chunks)
        self.vector_index.clear()
        self.vector_index.add_chunks(chunks)
        logger.info(f"[HybridRetriever] Indexed {len(chunks)} chunks across BM25 and Vector indices.")

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Executes hybrid retrieval:
        1. BM25 search for exact keywords, metrics, model names.
        2. Vector search for semantic and conceptual alignment.
        3. Reciprocal Rank Fusion (RRF) to merge and rerank results.
        """
        # Search twice the top_k from each retriever to allow healthy candidate overlap
        candidate_k = max(top_k * 2, 10)

        bm25_results = self.bm25_index.search(query, top_k=candidate_k)
        vector_results = self.vector_index.search(query, top_k=candidate_k)

        # Map candidate chunks by chunk_id
        chunk_map: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}
        bm25_ranks: Dict[str, int] = {}
        bm25_raw_scores: Dict[str, float] = {}
        vector_ranks: Dict[str, int] = {}
        vector_raw_scores: Dict[str, float] = {}

        # 1. Process BM25 rankings
        for rank, (chunk, score) in enumerate(bm25_results, start=1):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            bm25_ranks[cid] = rank
            bm25_raw_scores[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (self.bm25_weight / (self.rrf_k + rank))

        # 2. Process Vector rankings
        for rank, (chunk, score) in enumerate(vector_results, start=1):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            vector_ranks[cid] = rank
            vector_raw_scores[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (self.vector_weight / (self.rrf_k + rank))

        # 3. Sort by combined RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        results: List[RetrievedChunk] = []
        for cid in sorted_cids[:top_k]:
            c = chunk_map[cid]
            results.append(RetrievedChunk(
                chunk_id=cid,
                doc_id=c.get("doc_id", ""),
                text=c.get("text", ""),
                page=c.get("page"),
                section=c.get("section"),
                char_start=c.get("char_start", 0),
                char_end=c.get("char_end", 0),
                score=rrf_scores[cid],
                bm25_score=bm25_raw_scores.get(cid),
                vector_score=vector_raw_scores.get(cid),
                bm25_rank=bm25_ranks.get(cid),
                vector_rank=vector_ranks.get(cid),
                metadata={
                    "source_id": c.get("source_id"),
                    "url": c.get("url"),
                    "source_title": c.get("source_title")
                }
            ))

        logger.info(f"[HybridRetriever] Retrieved top {len(results)} chunks for query: '{query}'")
        return results
