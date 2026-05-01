from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.models import TaskStatus


def seed_task(client: TestClient, task):
    client.fake_store["tasks"][task.task_id] = task
    return task.task_id


def test_status_unknown_task_returns_404(client: TestClient):
    response = client.get("/tasks/non-existent-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"


def test_status_pending_task(client: TestClient):
    task = SimpleNamespace(
        task_id="task-pending",
        status=TaskStatus.PENDING,
        progress=0,
        stage="",
        message="",
        current_module=None,
        error=None,
        result_files={},
    )
    task_id = seed_task(client, task)

    response = client.get(f"/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"
    assert data["progress"] == 0
    assert data["error"] is None


def test_status_running_task(client: TestClient):
    task = SimpleNamespace(
        task_id="task-running",
        status=TaskStatus.RUNNING,
        progress=47,
        stage="M3:WhisperXASR",
        message="Transcribing audio...",
        current_module=None,
        error=None,
        result_files={},
    )
    task_id = seed_task(client, task)

    response = client.get(f"/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["progress"] == 47
    assert data["stage"] == "M3:WhisperXASR"
    assert data["message"] == "Transcribing audio..."


def test_status_failed_task(client: TestClient):
    task = SimpleNamespace(
        task_id="task-failed",
        status=TaskStatus.FAILED,
        progress=30,
        stage="M3:WhisperXASR",
        message="Pipeline error",
        current_module=None,
        error="WhisperX error: CUDA OOM",
        result_files={},
    )
    task_id = seed_task(client, task)

    response = client.get(f"/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert "WhisperX error" in data["error"]