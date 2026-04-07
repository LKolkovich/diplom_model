"""Tests for M1 – Audio Validator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.services.m1_audio_validator import validate_and_convert


def test_converts_wav_to_16k_mono(sample_video: Path, tmp_path: Path) -> None:
    out = validate_and_convert(sample_video, tmp_path)
    assert out.exists()
    assert out.suffix == ".wav"
    data, sr = sf.read(str(out))
    assert sr == 16_000
    assert data.ndim == 1


def test_accepts_existing_wav(silent_wav: Path, tmp_path: Path) -> None:
    out = validate_and_convert(silent_wav, tmp_path)
    assert out.exists()
    _, sr = sf.read(str(out))
    assert sr == 16_000


def test_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_and_convert(tmp_path / "nonexistent.mp4", tmp_path)
