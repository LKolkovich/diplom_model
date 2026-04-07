"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Generator, List

import cv2
import numpy as np
import pytest
import soundfile as sf


# ---------------------------------------------------------------------------
# Audio fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    """A 2-second 16 kHz mono silent WAV file."""
    samples = np.zeros(32_000, dtype=np.int16)
    path = tmp_path / "silent.wav"
    sf.write(str(path), samples, 16_000, subtype="PCM_16")
    return path


@pytest.fixture
def noisy_wav(tmp_path: Path) -> Path:
    """A 2-second 16 kHz mono noisy WAV file."""
    rng = np.random.default_rng(42)
    samples = (rng.standard_normal(32_000) * 0.1 * 32767).astype(np.int16)
    path = tmp_path / "noisy.wav"
    sf.write(str(path), samples, 16_000, subtype="PCM_16")
    return path


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """A tiny synthetic 3-second video at 30 fps (silent audio track)."""
    out = tmp_path / "sample.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=blue:size=640x360:rate=30:duration=3",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(out),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError:
        pytest.skip("ffmpeg is not installed")
    if result.returncode != 0:
        pytest.skip("ffmpeg failed to create test video")
    return out


# ---------------------------------------------------------------------------
# CV fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((200, 100, 3), dtype=np.uint8)


@pytest.fixture
def agent_template_dir(tmp_path: Path) -> Path:
    """Creates a small synthetic agent template library."""
    tdir = tmp_path / "agents"
    tdir.mkdir()
    rng = np.random.default_rng(0)
    for agent in ("jett", "reyna", "sage"):
        img = (rng.integers(0, 256, (36, 36, 3), dtype=np.uint8))
        cv2.imwrite(str(tdir / f"{agent}.png"), img)
    return tdir
