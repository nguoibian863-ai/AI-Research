import abc
import hashlib
import logging
import math
from typing import List, Optional
import httpx
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingBackend(abc.ABC):
    """Abstract Base Class for text embeddings."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        pass

    @abc.abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


class LocalHashEmbeddingBackend(EmbeddingBackend):
    """
    Ultra-lightweight deterministic CPU embedding model based on feature hashing (word + character n-grams).
    Zero external network calls, zero GPU VRAM consumption (ideal for 4GB VRAM constraint).
    Vectors are L2-normalized on the unit sphere for inner product / cosine similarity.
    """

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self._dim

        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = text.lower().split()

        # 1. Unigrams
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self._dim
            vec[h] += 1.0

        # 2. Bigrams
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            h = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16) % self._dim
            vec[h] += 1.5

        # 3. Subwords (3-4 char shingles for typo and morphological matching)
        for tok in tokens:
            if len(tok) >= 4:
                for k in range(len(tok) - 2):
                    shingle = tok[k:k+3]
                    h = int(hashlib.md5(shingle.encode("utf-8")).hexdigest(), 16) % self._dim
                    vec[h] += 0.5

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec.tolist()


class OllamaEmbeddingBackend(EmbeddingBackend):
    """
    Embedding backend delegating to a local Ollama instance (e.g. nomic-embed-text or bge-small).
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "nomic-embed-text",
        dimension: int = 768,
        timeout: float = 30.0
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dim = dimension
        self.timeout = timeout

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text}
                )
                res.raise_for_status()
                emb = res.json().get("embedding", [])
                if emb:
                    arr = np.array(emb, dtype=np.float32)
                    norm = np.linalg.norm(arr)
                    if norm > 1e-6:
                        arr /= norm
                    return arr.tolist()
        except Exception as e:
            logger.warning(f"[OllamaEmbeddingBackend] Ollama embed failed ({e}), falling back to LocalHashEmbedding.")

        # Fallback to local hash embedding if Ollama embedding is unreachable
        return LocalHashEmbeddingBackend(dimension=self._dim).embed_text(text)


class MockEmbeddingBackend(EmbeddingBackend):
    """
    Deterministic mock embedding backend for unit testing.
    """

    def __init__(self, dimension: int = 128):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        np.random.seed(abs(hash(text)) % (2**31 - 1))
        vec = np.random.randn(self._dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec.tolist()
