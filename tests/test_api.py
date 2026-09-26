import logging
import backend.main


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "llm_provider" in data
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
