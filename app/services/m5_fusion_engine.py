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
from app.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_MERGE_GAP = 1.5


@dataclass
class SpeakingInterval:
    agent: str
    start: float
    end: float
    avg_confidence: float = 0.0


def _build_intervals(
    detections: List[CVDetection],
    merge_gap: float = DEFAULT_MERGE_GAP,
) -> List[SpeakingInterval]:
    """Convert point-in-time CV detections into continuous intervals."""
    if not detections:
        return []

    agent_groups = defaultdict(list)
    for det in detections:
        agent_groups[det.agent_name].append(det)

    all_intervals: List[SpeakingInterval] = []

    for agent_name, dets in agent_groups.items():
        sorted_det = sorted(dets, key=lambda d: d.timestamp)
        current: Optional[List[CVDetection]] = None
        
        for det in sorted_det:
            if current is None:
                current = [det]
            elif (det.timestamp - current[-1].timestamp) <= merge_gap:
                current.append(det)
            else:
                avg_conf = sum(d.confidence for d in current) / len(current)
                all_intervals.append(SpeakingInterval(
                    agent=agent_name, 
                    start=current[0].timestamp, 
                    end=current[-1].timestamp,
                    avg_confidence=avg_conf
                ))
                current = [det]
        
        if current:
            avg_conf = sum(d.confidence for d in current) / len(current)
            all_intervals.append(SpeakingInterval(
                agent=agent_name, 
                start=current[0].timestamp, 
                end=current[-1].timestamp,
                avg_confidence=avg_conf
            ))

    all_intervals.sort(key=lambda x: x.start)
    logger.info("M5 – built %d speaking intervals from %d CV detections", len(all_intervals), len(detections))
    return all_intervals


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _score_agents_for_segment(
    seg: ASRSegment,
    intervals: List[SpeakingInterval],
) -> Dict[str, Tuple[float, float]]:
    # agent -> (total_overlap, sum_weighted_confidence)
    scores: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
    for iv in intervals:
        ov = _overlap(seg.start, seg.end, iv.start, iv.end)
        if ov > 0:
            scores[iv.agent][0] += ov
            scores[iv.agent][1] += ov * iv.avg_confidence
    
    # agent -> (total_overlap, avg_confidence)
    final_scores = {}
    for agent, (ov, weighted_conf) in scores.items():
        final_scores[agent] = (ov, weighted_conf / ov if ov > 0 else 0.0)
    return final_scores


def _build_speaker_map(
    segments: List[ASRSegment],
    intervals: List[SpeakingInterval],
    settings: Settings,
) -> Dict[str, str]:
    """Map generic speaker labels to agent names based on temporal coverage.

    Coverage is calculated as (total overlap with agent) / (total speaker duration).
    """
    speaker_durations: Dict[str, float] = defaultdict(float)
    # speaker -> agent -> (total_overlap, sum_weighted_confidence)
    speaker_agent_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))

    for seg in segments:
        duration = seg.end - seg.start
        if duration <= 0:
            continue
        speaker_durations[seg.speaker] += duration
        scores = _score_agents_for_segment(seg, intervals)
        for agent, (overlap, avg_conf) in scores.items():
            speaker_agent_stats[seg.speaker][agent][0] += overlap
            speaker_agent_stats[seg.speaker][agent][1] += overlap * avg_conf

    speaker_map: Dict[str, str] = {}
    for speaker, total_duration in speaker_durations.items():
        if speaker not in speaker_agent_stats:
            continue

        # Calculate coverage and average confidence for each agent
        candidates = []
        for agent, stats in speaker_agent_stats[speaker].items():
            overlap = stats[0]
            avg_conf = stats[1] / overlap if overlap > 0 else 0.0
            coverage = overlap / total_duration
            
            if coverage >= settings.fusion_coverage_threshold and avg_conf >= settings.fusion_confidence_threshold:
                candidates.append((agent, coverage, avg_conf))

        if len(candidates) == 1:
            best_agent, best_coverage, best_conf = candidates[0]
            speaker_map[speaker] = best_agent
            logger.info(
                "M5 – Speaker %s strong match: %s (coverage: %.2f%%, conf: %.2f)",
                speaker, best_agent, best_coverage * 100, best_conf
            )
        elif len(candidates) > 1:
            logger.warning(
                "M5 – Speaker %s has multiple strong candidates: %s. Mapping rejected due to ambiguity.",
                speaker, [c[0] for c in candidates]
            )
        else:
            # Fallback to old behavior but with warning? 
            # Or just don't map if no strong evidence.
            # The plan says: "Assign agent ONLY if exactly one candidate meets..."
            logger.info("M5 – Speaker %s has no strong CV evidence.", speaker)

    return speaker_map


def fuse(
    asr_segments: List[ASRSegment],
    cv_detections: List[CVDetection],
    settings: Settings,
    merge_gap: float = DEFAULT_MERGE_GAP,
) -> List[FusedSegment]:
    """Fuse ASR and CV data into speaker-labelled transcript segments.

    Parameters
    ----------
    asr_segments:
        Output of M3 – timestamped transcript segments with generic speaker IDs.
    cv_detections:
        Output of M4 – per-frame detections of which agent is speaking.
    settings:
        Application settings including fusion thresholds.
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
    speaker_map = _build_speaker_map(asr_segments, intervals, settings)

    fused: List[FusedSegment] = []
    for seg in asr_segments:
        resolved = speaker_map.get(seg.speaker, seg.speaker)
        fused.append(
            FusedSegment(start=seg.start, end=seg.end, speaker=resolved, text=seg.text)
        )

    return fused
