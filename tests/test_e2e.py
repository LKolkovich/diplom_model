"""End-to-end pipeline test.

Uses a synthetic video + silence to exercise the full M1→M6 pipeline without
requiring GPU/WhisperX. WhisperX (M3) is mocked so the test can run on CI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

from app.config import Settings
from app.models import ASRSegment, ProcessRequest
from app.orchestrator import create_task, get_task, run_pipeline
from app.models import TaskStatus


def _make_synthetic_video(path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=blue:size=640x360:rate=10:duration=4",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError:
        pytest.skip("ffmpeg is not installed")
    if result.returncode != 0:
        pytest.skip("ffmpeg unavailable – cannot create synthetic video")


def _fake_asr_segments() -> List[ASRSegment]:
    return [
        ASRSegment(start=0.5, end=1.5, text="rush B no stop", speaker="SPEAKER_00"),
        ASRSegment(start=2.0, end=3.0, text="I'll flank them", speaker="SPEAKER_01"),
    ]


def test_full_pipeline_no_gpu(tmp_path: Path) -> None:
    video = tmp_path / "game.mp4"
    _make_synthetic_video(video)

    settings = Settings(
        whisper_model="tiny",
        device="cpu",
        compute_type="int8",
        frame_rate=2,
        output_dir=tmp_path / "output",
        templates_dir=tmp_path / "agents",
        voicechat_roi="0.0,0.0,0.1,1.0",
        phash_threshold=10,
    )

    request = ProcessRequest(video_path=str(video))
    task_id = create_task()

    with patch(
        "app.services.m3_whisperx_asr.transcribe",
        return_value=_fake_asr_segments(),
    ):
        run_pipeline(task_id, request, settings)

    task = get_task(task_id)
    assert task is not None
    assert task.status == TaskStatus.completed, f"Pipeline failed: {task.error}"
    assert "combined" in task.result_files

    combined_srt = Path(task.result_files["combined"])
    assert combined_srt.exists()
    content = combined_srt.read_text()
    assert "rush B no stop" in content or "flank" in content


def test_pipeline_fails_gracefully_on_bad_video(tmp_path: Path) -> None:
    bad_video = tmp_path / "bad.mp4"
    bad_video.write_bytes(b"not a real video file")

    settings = Settings(
        whisper_model="tiny",
        device="cpu",
        compute_type="int8",
        output_dir=tmp_path / "output",
        templates_dir=tmp_path / "agents",
        voicechat_roi="0.0,0.0,0.1,1.0",
    )
    request = ProcessRequest(video_path=str(bad_video))
    task_id = create_task()
    run_pipeline(task_id, request, settings)

    task = get_task(task_id)
    assert task is not None
    assert task.status == TaskStatus.failed
    assert task.error is not None
