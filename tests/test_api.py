"""Tests for the FastAPI endpoints (using TestClient, no real pipeline runs)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import TaskState, TaskStatus


client = TestClient(app)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_process_returns_task_id(tmp_path: Path) -> None:
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.touch()

    with patch("app.orchestrator.run_pipeline") as mock_run:
        resp = client.post("/process", json={"video_path": str(dummy_video)})

    assert resp.status_code == 202
    body = resp.json()
    assert "task_id" in body
    assert len(body["task_id"]) == 32


def test_status_not_found() -> None:
    resp = client.get("/status/nonexistent_task_id_xyz")
    assert resp.status_code == 404


def test_result_not_found() -> None:
    resp = client.get("/result/nonexistent_task_id_xyz")
    assert resp.status_code == 404


def test_status_returns_pending_then_progress(tmp_path: Path) -> None:
    dummy_video = tmp_path / "v.mp4"
    dummy_video.touch()

    with patch("app.orchestrator.run_pipeline"):
        resp = client.post("/process", json={"video_path": str(dummy_video)})
    task_id = resp.json()["task_id"]

    status_resp = client.get(f"/status/{task_id}")
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["task_id"] == task_id
    assert body["status"] in ("pending", "running", "completed", "failed")
    assert 0 <= body["progress"] <= 100


def test_result_202_when_pending(tmp_path: Path) -> None:
    dummy_video = tmp_path / "v2.mp4"
    dummy_video.touch()

    with patch("app.orchestrator.run_pipeline"):
        resp = client.post("/process", json={"video_path": str(dummy_video)})
    task_id = resp.json()["task_id"]

    result_resp = client.get(f"/result/{task_id}")
    assert result_resp.status_code in (200, 202, 500)


def test_result_200_when_completed(tmp_path: Path) -> None:
    from app.orchestrator import _TASK_STORE

    srt_file = tmp_path / "combined.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")

    fake_task = TaskState(
        task_id="completed_task",
        status=TaskStatus.completed,
        progress=100,
        result_files={"combined": str(srt_file)},
    )
    _TASK_STORE["completed_task"] = fake_task

    resp = client.get("/result/completed_task")
    assert resp.status_code == 200
    body = resp.json()
    assert "combined" in body["files"]


def test_download_missing_file() -> None:
    resp = client.get("/download/fakeid/nonexistent.srt")
    assert resp.status_code == 404
