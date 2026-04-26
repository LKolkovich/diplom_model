"""Tests for M5 – Fusion Engine."""

from __future__ import annotations

from app.models import ASRSegment, CVDetection
from app.services.m5_fusion_engine import _build_intervals, fuse
from app.config import Settings


def _seg(start: float, end: float, speaker: str, text: str = "hello") -> ASRSegment:
    return ASRSegment(start=start, end=end, text=text, speaker=speaker)


def _det(ts: float, agent_name: str, conf: float = 0.9) -> CVDetection:
    return CVDetection(timestamp=ts, agent_name=agent_name, confidence=conf)


def test_build_intervals_merges_same_agent() -> None:
    dets = [_det(1.0, "jett"), _det(1.5, "jett"), _det(3.0, "jett")]
    intervals = _build_intervals(dets, merge_gap=2.0)
    assert len(intervals) == 1
    assert intervals[0].agent == "jett"
    assert intervals[0].start == 1.0
    assert intervals[0].end == 3.0


def test_build_intervals_splits_on_gap() -> None:
    dets = [_det(1.0, "jett"), _det(5.0, "jett")]
    intervals = _build_intervals(dets, merge_gap=1.0)
    assert len(intervals) == 2


def test_build_intervals_different_agents() -> None:
    dets = [_det(1.0, "jett"), _det(1.2, "reyna")]
    intervals = _build_intervals(dets, merge_gap=0.5)
    assert len(intervals) == 2
    agents = {iv.agent for iv in intervals}
    assert agents == {"jett", "reyna"}


def test_fuse_maps_speaker_to_agent() -> None:
    segs = [_seg(1.0, 3.0, "SPEAKER_00", "rush b")]
    dets = [_det(1.5, "jett"), _det(2.0, "jett")]
    settings = Settings()
    # Ensure thresholds are low enough for test data
    settings.fusion_coverage_threshold = 0.1
    settings.fusion_confidence_threshold = 0.1
    
    result = fuse(segs, dets, settings=settings)
    assert len(result) == 1
    assert result[0].speaker == "jett"
    assert result[0].text == "rush b"


def test_fuse_no_cv_keeps_original_labels() -> None:
    segs = [_seg(1.0, 3.0, "SPEAKER_00")]
    settings = Settings()
    result = fuse(segs, [], settings=settings)
    assert result[0].speaker == "SPEAKER_00"


def test_fuse_multiple_speakers() -> None:
    segs = [
        _seg(0.0, 2.0, "SPEAKER_00"),
        _seg(3.0, 5.0, "SPEAKER_01"),
    ]
    dets = [
        _det(0.5, "jett"), _det(1.0, "jett"),
        _det(3.5, "reyna"), _det(4.0, "reyna"),
    ]
    settings = Settings()
    settings.fusion_coverage_threshold = 0.1
    settings.fusion_confidence_threshold = 0.1
    
    result = fuse(segs, dets, settings=settings)
    speakers = {s.speaker for s in result}
    assert "jett" in speakers
    assert "reyna" in speakers
