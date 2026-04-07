"""M1 – Audio Validator.

Accepts a video or audio file and produces a 16 kHz, mono, 16-bit PCM WAV
file suitable for WhisperX ingestion.  Conversion is delegated entirely to
ffmpeg so we avoid loading the full audio into memory.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

TARGET_SR = 16_000
TARGET_CHANNELS = 1
TARGET_FORMAT = "s16"


def validate_and_convert(source_path: str | Path, output_dir: str | Path) -> Path:
    """Convert *source_path* to a 16 kHz mono WAV file.

    Parameters
    ----------
    source_path:
        Path to the source video or audio file.
    output_dir:
        Directory where the converted WAV will be written.

    Returns
    -------
    Path
        Absolute path of the converted WAV file.

    Raises
    ------
    FileNotFoundError
        When *source_path* does not exist.
    RuntimeError
        When ffmpeg conversion fails.
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_path = out_dir / f"{source.stem}_16k.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(source),
        "-vn",
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SR),
        "-sample_fmt", TARGET_FORMAT,
        "-f", "wav",
        str(wav_path),
    ]

    logger.info("M1 – converting audio: %s → %s", source, wav_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg executable not found. Install ffmpeg and ensure it is on PATH."
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}"
        )

    if not wav_path.exists() or wav_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced an empty or missing output file")

    logger.info("M1 – audio ready: %s (%.1f MB)", wav_path, wav_path.stat().st_size / 1e6)
    return wav_path
