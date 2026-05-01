from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient):
    response = client.get("/tasks/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_200(client: TestClient):
    response = client.get("/tasks/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["pipeline_busy"], bool)