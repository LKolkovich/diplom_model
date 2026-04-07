"""M5 – Fusion Engine.

Combines WhisperX ASR segments (M3) with CV-detected speaking intervals (M4)
to map generic speaker IDs ("SPEAKER_00") to Valorant agent names ("Jett").

Algorithm
---------
1. Convert the list of CVDetection events into continuous speaking *intervals*
   per agent by merging detections that are close in time (gap ≤ ``merge_gap``
   seconds).
2. For each ASR segment, score every agent by computing the total temporal
   overlap between the segment's [start, end] window and that agent's
   intervals.
3. Assign the agent with the highest overlap to that segment's speaker label.
4. Build a stable speaker→agent mapping by majority vote per speaker label,
   then apply the mapping to all segments.
5. Segments with no CV evidence keep their original speaker label.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.models import ASRSegment, CVDetection, FusedSegment

logger = logging.getLogger(__name__)

DEFAULT_MERGE_GAP = 1.5


@dataclass
class SpeakingInterval:
    agent: str
    start: float
    end: float


def _build_intervals(
    detections: List[CVDetection],
    merge_gap: float = DEFAULT_MERGE_GAP,
) -> List[SpeakingInterval]:
    """Convert point-in-time CV detections into continuous intervals."""
    if not detections:
        return []

    sorted_det = sorted(detections, key=lambda d: d.timestamp)
    intervals: List[SpeakingInterval] = []

    current: Optional[SpeakingInterval] = None
    for det in sorted_det:
        if current is None:
            current = SpeakingInterval(agent=det.agent, start=det.timestamp, end=det.timestamp)
        elif det.agent == current.agent and (det.timestamp - current.end) <= merge_gap:
            current.end = det.timestamp
        else:
            intervals.append(current)
            current = SpeakingInterval(agent=det.agent, start=det.timestamp, end=det.timestamp)

    if current is not None:
        intervals.append(current)

    logger.info("M5 – built %d speaking intervals from %d CV detections", len(intervals), len(detections))
    return intervals


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _score_agents_for_segment(
    seg: ASRSegment,
    intervals: List[SpeakingInterval],
) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    for iv in intervals:
        ov = _overlap(seg.start, seg.end, iv.start, iv.end)
        if ov > 0:
            scores[iv.agent] += ov
    return scores


def _build_speaker_map(
    segments: List[ASRSegment],
    intervals: List[SpeakingInterval],
) -> Dict[str, str]:
    """Map each generic speaker label to an agent name by majority vote."""
    votes: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for seg in segments:
        scores = _score_agents_for_segment(seg, intervals)
        if scores:
            best_agent = max(scores, key=lambda k: scores[k])
            votes[seg.speaker][best_agent] += scores[best_agent]

    speaker_map: Dict[str, str] = {}
    for speaker, agent_scores in votes.items():
        if agent_scores:
            speaker_map[speaker] = max(agent_scores, key=lambda k: agent_scores[k])

    logger.info("M5 – speaker mapping: %s", speaker_map)
    return speaker_map


def fuse(
    asr_segments: List[ASRSegment],
    cv_detections: List[CVDetection],
    merge_gap: float = DEFAULT_MERGE_GAP,
) -> List[FusedSegment]:
    """Fuse ASR and CV data into speaker-labelled transcript segments.

    Parameters
    ----------
    asr_segments:
        Output of M3 – timestamped transcript segments with generic speaker IDs.
    cv_detections:
        Output of M4 – per-frame detections of which agent is speaking.
    merge_gap:
        Maximum gap (seconds) between consecutive same-agent detections
        before they are treated as separate intervals.

    Returns
    -------
    list[FusedSegment]
        ASR segments enriched with resolved agent-name speaker labels.
    """
    if not cv_detections:
        logger.warning("M5 – no CV detections; keeping original speaker labels.")
        return [
            FusedSegment(start=s.start, end=s.end, speaker=s.speaker, text=s.text)
            for s in asr_segments
        ]

    intervals = _build_intervals(cv_detections, merge_gap)
    speaker_map = _build_speaker_map(asr_segments, intervals)

    fused: List[FusedSegment] = []
    for seg in asr_segments:
        resolved = speaker_map.get(seg.speaker, seg.speaker)
        fused.append(
            FusedSegment(start=seg.start, end=seg.end, speaker=resolved, text=seg.text)
        )

    return fused
