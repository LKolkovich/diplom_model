import io
import uuid

from fastapi.testclient import TestClient


def make_video_file(name: str = "clip.mp4", size: int = 1024):
    return {
        "video": (name, io.BytesIO(b"0" * size), "video/mp4"),
    }


def test_submit_returns_202_with_task_id(client: TestClient):
    response = client.post("/tasks", files=make_video_file())
    assert response.status_code == 202

    data = response.json()
    assert data["status"] == "pending"
    assert data["message"] == "Task queued"
    uuid.UUID(data["task_id"])


def test_submit_without_video_returns_422(client: TestClient):
    response = client.post("/tasks", data={})
    assert response.status_code == 422


def test_submit_with_optional_params(client: TestClient):
    response = client.post(
        "/tasks",
        files=make_video_file(),
        data={
            "language": "ru",
            "min_speakers": "2",
            "max_speakers": "4",
            "initial_prompt": "agent comms",
            "video_mode": "user_crop",
            "roi": "10,20,300,400",
            "level_audio": "true",
            "compression_ratio": "3.0",
            "compression_threshold": "-18.0",
            "fps": "5",
            "agents_list": "clove,sage,skye",
            "hf_token": "secret",
        },
    )
    assert response.status_code == 202
    assert response.json()["status"] == "pending"


def test_submit_invalid_video_mode_returns_422(client: TestClient):
    response = client.post(
        "/tasks",
        files=make_video_file(),
        data={"video_mode": "wrong-mode"},
    )
    assert response.status_code == 422


def test_submit_saves_video_to_tmp(client: TestClient, fake_settings):
    response = client.post("/tasks", files=make_video_file(name="input.mp4", size=512))
    task_id = response.json()["task_id"]

    saved_file = fake_settings.tmp_dir / task_id / "input.mp4"
    assert saved_file.exists()
    assert saved_file.stat().st_size == 512