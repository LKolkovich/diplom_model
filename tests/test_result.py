from pathlib import Path
from types import SimpleNamespace
import zipfile
import io

from fastapi.testclient import TestClient

from app.models import TaskStatus


def seed_task(client: TestClient, task):
    client.fake_store["tasks"][task.task_id] = task
    return task.task_id


def test_result_unknown_task_returns_404(client: TestClient):
    response = client.get("/tasks/unknown/result")
    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"


def test_result_pending_returns_202(client: TestClient):
    task = SimpleNamespace(
        task_id="task-pending",
        status=TaskStatus.RUNNING,
        progress=47,
        stage="M3:WhisperXASR",
        message="Transcribing audio...",
        current_module=None,
        error=None,
        result_files={},
    )
    task_id = seed_task(client, task)

    response = client.get(f"/tasks/{task_id}/result")
    assert response.status_code == 202
    data = response.json()
    assert data["task_id"] == task_id
    assert data["status"] == "running"
    assert data["progress"] == 47


def test_result_failed_returns_500(client: TestClient):
    task = SimpleNamespace(
        task_id="task-failed",
        status=TaskStatus.FAILED,
        progress=33,
        stage="M3:WhisperXASR",
        message="Pipeline error",
        current_module=None,
        error="WhisperX error: CUDA OOM",
        result_files={},
    )
    task_id = seed_task(client, task)

    response = client.get(f"/tasks/{task_id}/result")
    assert response.status_code == 500
    data = response.json()
    assert data["task_id"] == task_id
    assert data["status"] == "failed"
    assert "WhisperX error" in data["error"]


def test_result_completed_returns_zip(client: TestClient, tmp_path: Path):
    clove = tmp_path / "clove.srt"
    sage = tmp_path / "sage.srt"
    clove.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    sage.write_text("1\n00:00:00,000 --> 00:00:01,000\nworld\n", encoding="utf-8")

    task = SimpleNamespace(
        task_id="task-completed",
        status=TaskStatus.COMPLETED,
        progress=100,
        stage="M6:Done",
        message="Done",
        current_module=None,
        error=None,
        result_files={
            "clove": str(clove),
            "sage": str(sage),
        },
    )
    task_id = seed_task(client, task)

    response = client.get(f"/tasks/{task_id}/result")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert f'result_{task_id}.zip' in response.headers["content-disposition"]

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = sorted(archive.namelist())
    assert names == ["subtitles/clove.srt", "subtitles/sage.srt"]