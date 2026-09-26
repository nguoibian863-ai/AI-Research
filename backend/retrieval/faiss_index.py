import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss

from backend.retrieval.embeddings import EmbeddingBackend, LocalHashEmbeddingBackend
from backend.db.database import DatabaseManager
from backend.db.repositories import ChunkRepository

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path("data/indexes")


class FaissVectorIndex:
    """
    CPU-optimized FAISS vector index using Inner Product (Cosine Similarity on L2-normalized embeddings).
    Stores vector embeddings in FAISS while keeping metadata alongside.
    Can be rebuilt completely from SQLite (Source of Truth) at any time.
    """

    def __init__(
        self,
        embedding_backend: Optional[EmbeddingBackend] = None,
        index_dir: Path | str = DEFAULT_INDEX_DIR
    ):
        self.embedding = embedding_backend or LocalHashEmbeddingBackend()
        self.dimension = self.embedding.dimension
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.chunks_meta: List[Dict[str, Any]] = []

    def clear(self) -> None:
        """Resets the vector index and metadata."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_meta = []

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Adds a list of chunks with metadata into FAISS."""
        valid_chunks = [c for c in chunks if c.get("text", "").strip()]
        if not valid_chunks:
            return

        texts = [c["text"] for c in valid_chunks]
        embeddings = self.embedding.embed_batch(texts)
        emb_arr = np.array(embeddings, dtype=np.float32)

        # Ensure L2 normalized
        faiss.normalize_L2(emb_arr)

        self.index.add(emb_arr)
        self.chunks_meta.extend(valid_chunks)
        logger.info(f"[FaissVectorIndex] Added {len(valid_chunks)} chunks to vector index (Total: {self.index.ntotal}).")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches the index with semantic query vector.
        Returns top_k (chunk_dict, score) tuples sorted descending by score.
        """
        if self.index.ntotal == 0 or not query.strip():
            return []

        q_vec = np.array([self.embedding.embed_text(query)], dtype=np.float32)
        faiss.normalize_L2(q_vec)

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q_vec, k)

        results: List[Tuple[Dict[str, Any], float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks_meta):
                results.append((self.chunks_meta[idx], float(dist)))

        return results

    def save(self, index_name: str) -> None:
        """Persists the FAISS index binary and metadata JSON to disk."""
        faiss_path = self.index_dir / f"{index_name}.faiss"
        meta_path = self.index_dir / f"{index_name}_meta.json"

        faiss.write_index(self.index, str(faiss_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks_meta, f, ensure_ascii=False, indent=2)
        logger.info(f"[FaissVectorIndex] Saved index '{index_name}' with {self.index.ntotal} items to {self.index_dir}.")

    def load(self, index_name: str) -> bool:
        """Loads index and metadata from disk if they exist."""
        faiss_path = self.index_dir / f"{index_name}.faiss"
        meta_path = self.index_dir / f"{index_name}_meta.json"

        if not faiss_path.exists() or not meta_path.exists():
            return False

        try:
            self.index = faiss.read_index(str(faiss_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                self.chunks_meta = json.load(f)
            logger.info(f"[FaissVectorIndex] Loaded index '{index_name}' ({self.index.ntotal} vectors).")
            return True
        except Exception as e:
            logger.warning(f"[FaissVectorIndex] Failed to load index '{index_name}': {e}")
            return False

    def rebuild_from_db(self, db: DatabaseManager, session_id: str) -> int:
        """
        P0 Reliability: Rebuilds vector index from SQLite Source of Truth if corrupted or missing.
        """
        logger.info(f"[FaissVectorIndex] Rebuilding index for session {session_id} from SQLite Source of Truth...")
        self.clear()
        repo = ChunkRepository(db)
        chunks = repo.get_by_session(session_id)
        if chunks:
            self.add_chunks(chunks)
            self.save(session_id)
        return len(chunks)
