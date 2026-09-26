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
