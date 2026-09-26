import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app
from backend.api.research import get_engine
from backend.core.engine import ResearchEngine
from backend.core.limits import ResearchLimits
from backend.db.database import DatabaseManager
from backend.llm.mock import MockLLMBackend
from backend.tools.web_search import SearchResultItem
from backend.tools.web_fetch import FetchedWebContent


class FakeSearchTool:
    def __init__(self, items=None):
        self.called_queries = []
        self.items = items

    def search(self, query: str, max_results: int = 8):
        self.called_queries.append(query)
        if self.items is not None:
            return self.items[:max_results]
        return [
            SearchResultItem(
                title=f"Benchmark and Leaderboard: {query}",
                url="https://arxiv.org/abs/2006.11275",
                snippet=f"Detailed investigation for {query} with metrics and results.",
                query=query,
                rank=1
            ),
            SearchResultItem(
                title=f"Technical Report: {query}",
                url="https://arxiv.org/abs/1812.05784",
                snippet=f"Comprehensive evaluation and benchmarks for {query}.",
                query=query,
                rank=2
            )
        ]


class FakeFetchTool:
    def __init__(self, doc_text: str = None):
        self.called_urls = []
        self.doc_text = doc_text or (
            "CenterPoint: Center-based 3D Object Detection. "
            "CenterPoint achieves 60.3 mAP and 67.3 NDS on nuScenes benchmark with 3D LiDAR. "
            "PointPillars achieves 59.2 NDS with fast encoder."
        )

    def fetch(self, url: str):
        self.called_urls.append(url)
        return FetchedWebContent(
            url=url,
            title="Extracted Document Title",
            text=self.doc_text,
            content_hash=f"hash_{hash(url)}",
            is_cached=False
        )


@pytest.fixture
def isolated_engine(tmp_path: Path):
    """Provides a fresh isolated ResearchEngine with a temp SQLite database, fake tools, and fast offline embeddings."""
    from backend.retrieval.embeddings import LocalHashEmbeddingBackend

    db_file = tmp_path / "isolated_test.db"
    db = DatabaseManager(db_path=db_file)
    mock_llm = MockLLMBackend()
    search = FakeSearchTool()
    fetch = FakeFetchTool()

    engine = ResearchEngine(
        llm=mock_llm,
        db=db,
        limits=ResearchLimits(max_research_steps=3, max_search_calls=5, max_fetch_calls=5),
        search_tool=search,
        fetch_tool=fetch,
        embedding_backend=LocalHashEmbeddingBackend(dimension=128)
    )
    return engine


@pytest.fixture
def client(isolated_engine, tmp_path: Path, monkeypatch):
    """FastAPI TestClient with isolated engine injected via dependency override."""
    import backend.main
    monkeypatch.setattr(backend.main, "LOGS_DIR", tmp_path / "logs")
    app.dependency_overrides[get_engine] = lambda: isolated_engine
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
