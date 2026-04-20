"""Tests for M2 – Frame Extractor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.m2_frame_extractor import extract_frames_to_disk, iter_frames


ROI_FULL = (0.0, 0.0, 1.0, 1.0)
ROI_LEFT = (0.0, 0.0, 0.1, 1.0)


def test_iter_frames_yields_correct_roi(sample_video: Path) -> None:
    frames = list(iter_frames(sample_video, ROI_LEFT, target_fps=2))
    assert len(frames) > 0
    for ef in frames:
        assert ef.frame.ndim == 3
        assert ef.frame.shape[2] == 3
        assert ef.timestamp >= 0.0


def test_iter_frames_full_roi(sample_video: Path) -> None:
    frames = list(iter_frames(sample_video, ROI_FULL, target_fps=2))
    assert len(frames) >= 2


def test_extract_to_disk_creates_files(sample_video: Path, tmp_path: Path) -> None:
    paths = extract_frames_to_disk(sample_video, tmp_path, ROI_LEFT, target_fps=2)
    assert len(paths) > 0
    for p in paths:
        assert p.exists()
        assert p.suffix == ".jpg"


def test_raises_on_bad_video(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(RuntimeError, match="Cannot open video file"):
        list(iter_frames(bad, ROI_FULL))


def test_iter_frames_debug_saves_png(sample_video: Path, tmp_path: Path) -> None:
    debug_dir = tmp_path / "debug"
    frames = list(
        iter_frames(
            sample_video, ROI_LEFT, target_fps=2, debug_frames=True, debug_frames_dir=debug_dir
        )
    )
    assert len(frames) > 0
    pngs = list(debug_dir.glob("*.png"))
    assert len(pngs) == len(frames)
    for p in pngs:
        assert p.exists()
        assert p.suffix == ".png"
