import logging
import pytest
import backend.main


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "llm_provider" in data
    assert "embedding_backend" in data
    assert "database" in data


def test_create_session_endpoint(client):
    response = client.post("/api/research/session", json={"goal": "Investigate Sparse4D v3 latency"})
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["goal"] == "Investigate Sparse4D v3 latency"
    assert data["phase"] == "INIT"
    assert data["step"] == 0

    # Retrieve session
    session_id = data["session_id"]
    get_res = client.get(f"/api/research/session/{session_id}")
    assert get_res.status_code == 200
    sess_body = get_res.json()
    assert sess_body["session"]["session_id"] == session_id
    assert sess_body["session"]["goal"] == "Investigate Sparse4D v3 latency"


def test_list_trajectories_endpoint(client):
    response = client.get("/api/research/trajectories/gold")
    assert response.status_code == 200
    data = response.json()
    assert data["partition"] == "gold"
    assert "count" in data
    assert isinstance(data["samples"], list)


def test_health_uses_injected_engine(client, isolated_engine, monkeypatch):
    """/health must go through Depends(get_engine), never building the real engine/DB."""
    def _forbidden():
        raise AssertionError("get_research_engine() must not be called when get_engine is overridden")
    monkeypatch.setattr(backend.main, "get_research_engine", _forbidden)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["database"] == str(isolated_engine.db.db_path)
    assert response.json()["llm_provider"] == "mock"


def test_logging_configured_in_lifespan_only(isolated_engine, tmp_path, monkeypatch):
    """Importing backend.main must not touch the root logger; lifespan adds and removes its own handlers."""
    from fastapi.testclient import TestClient
    from backend.api.research import get_engine

    root = logging.getLogger()
    sentinel = logging.NullHandler()
    root.addHandler(sentinel)
    before = list(root.handlers)
    monkeypatch.setattr(backend.main, "LOGS_DIR", tmp_path / "logs")
    backend.main.app.dependency_overrides[get_engine] = lambda: isolated_engine
    try:
        with TestClient(backend.main.app):
            assert sentinel in root.handlers  # existing handlers are preserved
            assert len(root.handlers) == len(before) + 2
            assert (tmp_path / "logs" / "research.log").exists()
        assert root.handlers == before  # lifespan shutdown removes only its own handlers
    finally:
        backend.main.app.dependency_overrides.clear()
        root.removeHandler(sentinel)


def test_lifespan_eagerly_validates_engine_and_fails_on_broken_backend(monkeypatch, tmp_path):
    """
    P1 Test: Startup backend validation.
    Verifies that FastAPI lifespan eagerly instantiates the ResearchEngine to fail-fast
    at server startup if the embedding backend or database cannot be initialized.
    """
    from fastapi.testclient import TestClient

    monkeypatch.setattr(backend.main, "LOGS_DIR", tmp_path / "logs")

    def _broken_engine_factory():
        raise RuntimeError("Embedding model initialization failed: download error")

    monkeypatch.setattr(backend.main, "get_research_engine", _broken_engine_factory)

    backend.main.app.dependency_overrides.clear()

    with pytest.raises(RuntimeError, match="Embedding model initialization failed"):
        with TestClient(backend.main.app):
            pass


def test_show_session_script_renders_saved_session(tmp_path, monkeypatch):
    """scripts/show_session.py lets the owner inspect results without a server or SQLite GUI."""
    import importlib.util
    from pathlib import Path
    from backend.core.engine import ResearchEngine
    from backend.core.limits import ResearchLimits
    from backend.db.database import DatabaseManager
    from backend.llm.mock import MockLLMBackend
    from backend.retrieval.embeddings import LocalHashEmbeddingBackend
    from tests.conftest import FakeSearchTool, FakeFetchTool

    db = DatabaseManager(db_path=tmp_path / "show.db")
    canned = {
        "GeneratedQueriesSchema": {"queries": [{"query": "CenterPoint PointPillars nuScenes", "rationale": "r"}]},
        "ExtractedEvidencesSchema": {"facts": [
            {"statement": "CenterPoint achieves 67.3 NDS on nuScenes.", "subject": "CenterPoint", "predicate": "achieves",
             "metric": "NDS", "value": "67.3", "raw_quote": "CenterPoint achieves 60.3 mAP and 67.3 NDS"}]},
        "generate": "CenterPoint reaches 67.3 NDS [E1].",
    }
    engine = ResearchEngine(MockLLMBackend(canned), db, ResearchLimits(max_research_steps=1),
                            FakeSearchTool(), FakeFetchTool(), embedding_backend=LocalHashEmbeddingBackend(64))
    result = engine.run_week1("Compare CenterPoint and PointPillars on nuScenes")

    spec = importlib.util.spec_from_file_location("show_session", Path("scripts/show_session.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sessions = module.list_sessions(db)
    assert sessions[0]["session_id"] == result["session_id"]

    text = module.render_session(db, result["session_id"])
    assert "Compare CenterPoint and PointPillars on nuScenes" in text
    assert "CenterPoint achieves 60.3 mAP and 67.3 NDS" in text   # verbatim quote
    assert "## Claims trong report" in text
    assert "CenterPoint reaches 67.3 NDS [E1]" in text              # report body


def test_session_viewer_page_and_list_endpoint(client, isolated_engine):
    """Session Viewer (plan 29.0): /ui serves a static read-only page; /sessions lists recent sessions."""
    from pathlib import Path

    ui = client.get("/ui")
    assert ui.status_code == 200
    assert "text/html" in ui.headers["content-type"]
    assert "Research Session Viewer" in ui.text

    # Page must never inject fetched web content as HTML (XSS from scraped pages)
    page_source = Path("backend/ui/index.html").read_text(encoding="utf-8")
    assert "innerHTML" not in page_source
    assert "insertAdjacentHTML" not in page_source

    first = client.post("/api/research/session", json={"goal": "First goal"}).json()["session_id"]
    second = client.post("/api/research/session", json={"goal": "Second goal"}).json()["session_id"]

    listing = client.get("/api/research/sessions").json()
    ids = [s["session_id"] for s in listing["sessions"]]
    assert listing["count"] == 2
    assert ids[0] == second and ids[1] == first  # newest first
    assert {"goal", "phase", "status", "created_at"} <= set(listing["sessions"][0])

    detail = client.get(f"/api/research/session/{first}").json()
    assert "error_message" in detail["session"]
