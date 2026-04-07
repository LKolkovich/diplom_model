"""Tests for M6 – SRT Exporter."""

from __future__ import annotations

from pathlib import Path

from app.models import FusedSegment
from app.services.m6_srt_exporter import _format_timestamp, export


def test_format_timestamp_zero() -> None:
    assert _format_timestamp(0.0) == "00:00:00,000"


def test_format_timestamp_rounded() -> None:
    assert _format_timestamp(3661.5) == "01:01:01,500"


def test_format_timestamp_millis() -> None:
    assert _format_timestamp(1.123) == "00:00:01,123"


def test_export_creates_combined_and_per_speaker(tmp_path: Path) -> None:
    segments = [
        FusedSegment(start=0.0, end=2.0, speaker="jett", text="Rush B!"),
        FusedSegment(start=3.0, end=5.0, speaker="reyna", text="I'll flank."),
        FusedSegment(start=6.0, end=8.0, speaker="jett", text="OK go!"),
    ]
    files = export(segments, tmp_path, task_id="test123")

    assert "combined" in files
    assert "jett" in files
    assert "reyna" in files

    combined_text = Path(files["combined"]).read_text()
    assert "[jett]: Rush B!" in combined_text
    assert "[reyna]: I'll flank." in combined_text

    jett_text = Path(files["jett"]).read_text()
    assert "Rush B!" in jett_text
    assert "OK go!" in jett_text
    assert "[jett]" not in jett_text


def test_export_numbering_is_sequential(tmp_path: Path) -> None:
    segments = [
        FusedSegment(start=i * 2.0, end=i * 2.0 + 1.5, speaker="sage", text=f"Line {i}")
        for i in range(5)
    ]
    files = export(segments, tmp_path, task_id="seq")
    text = Path(files["sage"]).read_text()
    for n in range(1, 6):
        assert f"\n{n}\n" in text or text.startswith(f"{n}\n")


def test_export_empty_segments(tmp_path: Path) -> None:
    files = export([], tmp_path, task_id="empty")
    assert "combined" in files
    combined_text = Path(files["combined"]).read_text()
    assert combined_text.strip() == ""
