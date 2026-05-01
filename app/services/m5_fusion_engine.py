"""M5 – Fusion Engine.

Combines WhisperX ASR segments (M3) with CV-detected speaking intervals (M4)
to map generic speaker IDs ("SPEAKER_00") to Valorant agent names ("Jett").

Pipeline
--------
Phase 1  (global / hard matching)
  - Build speaker × agent coverage matrix from temporal overlap.
  - Derive hard (discrete) speaker↔agent pairs when evidence is unambiguous.
  - Derive forbidden pairs for clearly implausible matches.

Phase 2  (per-segment soft fusion)
  - For each segment combine three sources of evidence:
      * M3 global diarization map (coverage-based scores per speaker)
      * M3 speaker_candidates (alternative speaker hypotheses from ASR)
      * M4 per-segment CV overlap (who is visually active right now)
  - Formula: combined_score(A) = alpha * diar_score(A) + beta * cv_score(A)
  - Segments whose speaker is in hard_map get a fixed prior boost.
  - Stores all candidates + attribution metadata in FusedSegment.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from app.models import ASRSegment, CVDetection, FusedSegment, SpeakerAgentCandidate
from app.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_MERGE_GAP = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# Internal dataclasses (not exported; models.py owns the public types)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpeakingInterval:
    agent: str
    start: float
    end: float
    avg_confidence: float = 0.0


@dataclass
class SpeakerAgentEvidence:
    overlap: float = 0.0
    weighted_conf_sum: float = 0.0
    coverage: float = 0.0
    avg_confidence: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

# ─────────────────────────────────────────────────────────────────────────────
# Complete mapping helpers
# ─────────────────────────────────────────────────────────────────────────────

def _should_force_complete_mapping(
        speaker_durations: Dict[str, float],
        all_agents: Set[str],
) -> bool:
    """Enable full one-to-one speaker↔agent mapping when counts match."""
    unique_speakers = sorted(speaker_durations.keys())
    unique_agents = sorted(all_agents)
    return len(unique_speakers) > 0 and len(unique_speakers) == len(unique_agents)


def _build_complete_map_when_counts_match(
        speaker_durations: Dict[str, float],
        evidence: Dict[str, Dict[str, SpeakerAgentEvidence]],
        all_agents: Set[str],
        hard_map: Dict[str, str],
) -> Tuple[Dict[str, str], List[Tuple[str, str, float, float, bool]]]:
    """
    Build a complete one-to-one speaker -> agent map when counts match.

    Rules:
    - hard_map pairs are immutable
    - each remaining speaker gets exactly one remaining agent
    - each remaining agent is used at most once
    - primary ranking: coverage
    - secondary ranking: avg_confidence
    - pairs with zero evidence are allowed only as last-resort fallback
    """
    final_map: Dict[str, str] = dict(hard_map)
    fallback_assignments: List[Tuple[str, str, float, float, bool]] = []

    unique_speakers = sorted(speaker_durations.keys())
    unique_agents = sorted(all_agents)

    locked_speakers = set(hard_map.keys())
    locked_agents = set(hard_map.values())

    remaining_speakers = [s for s in unique_speakers if s not in locked_speakers]
    remaining_agents = [a for a in unique_agents if a not in locked_agents]

    if not remaining_speakers:
        return final_map, fallback_assignments

    scored_pairs: List[Tuple[float, float, str, str, bool]] = []
    for speaker in remaining_speakers:
        for agent in remaining_agents:
            cell = evidence.get(speaker, {}).get(agent)
            coverage = cell.coverage if cell else 0.0
            avg_conf = cell.avg_confidence if cell else 0.0
            zero_evidence = cell is None or (coverage <= 0.0 and avg_conf <= 0.0)
            scored_pairs.append((coverage, avg_conf, speaker, agent, zero_evidence))

    scored_pairs.sort(
        key=lambda x: (
            x[0],          # coverage
            x[1],          # avg_confidence
            0 if not x[4] else -1,  # prefer non-fallback
            x[2],          # deterministic speaker
            x[3],          # deterministic agent
        ),
        reverse=True,
    )

    assigned_speakers: Set[str] = set()
    assigned_agents: Set[str] = set()

    # Greedy assignment on evidence-backed pairs first
    for coverage, avg_conf, speaker, agent, zero_evidence in scored_pairs:
        if zero_evidence:
            continue
        if speaker in assigned_speakers or agent in assigned_agents:
            continue
        final_map[speaker] = agent
        assigned_speakers.add(speaker)
        assigned_agents.add(agent)

    # Residual fallback assignment for anything still unmatched
    residual_speakers = [s for s in remaining_speakers if s not in assigned_speakers]
    residual_agents = [a for a in remaining_agents if a not in assigned_agents]

    for speaker, agent in zip(sorted(residual_speakers), sorted(residual_agents)):
        cell = evidence.get(speaker, {}).get(agent)
        coverage = cell.coverage if cell else 0.0
        avg_conf = cell.avg_confidence if cell else 0.0
        zero_evidence = cell is None or (coverage <= 0.0 and avg_conf <= 0.0)
        final_map[speaker] = agent
        fallback_assignments.append((speaker, agent, coverage, avg_conf, zero_evidence))

    return final_map, fallback_assignments


def _apply_final_map_to_segment(
        seg: ASRSegment,
        winner: Optional[SpeakerAgentCandidate],
        final_map: Dict[str, str],
        counts_match: bool,
        winner_threshold: float,
) -> str:
    """
    Decide the final output label for a segment.

    counts_match=True:
      - if seg.speaker exists in final_map, always emit final_map[seg.speaker]
    counts_match=False:
      - preserve original speaker for unresolved / ambiguous cases
      - only emit winner.agent when the winner is sufficiently confident
    """
    if counts_match and seg.speaker in final_map:
        return final_map[seg.speaker]

    if winner is None:
        return seg.speaker

    if winner.score < winner_threshold:
        return seg.speaker

    return winner.agent

def _build_intervals(
    detections: List[CVDetection],
    merge_gap: float = DEFAULT_MERGE_GAP,
) -> List[SpeakingInterval]:
    """Convert point-in-time CV detections into continuous intervals."""
    if not detections:
        return []

    agent_groups: Dict[str, List[CVDetection]] = defaultdict(list)
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
                all_intervals.append(
                    SpeakingInterval(
                        agent=agent_name,
                        start=current[0].timestamp,
                        end=current[-1].timestamp,
                        avg_confidence=avg_conf,
                    )
                )
                current = [det]

        if current:
            avg_conf = sum(d.confidence for d in current) / len(current)
            all_intervals.append(
                SpeakingInterval(
                    agent=agent_name,
                    start=current[0].timestamp,
                    end=current[-1].timestamp,
                    avg_confidence=avg_conf,
                )
            )

    all_intervals.sort(key=lambda x: x.start)
    logger.info(
        "M5 – built %d speaking intervals from %d CV detections",
        len(all_intervals),
        len(detections),
    )
    return all_intervals


def _score_agents_for_segment(
    seg: ASRSegment,
    intervals: List[SpeakingInterval],
) -> Dict[str, Tuple[float, float]]:
    """Return per-agent (total_overlap, avg_confidence) for one ASR segment."""
    raw: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])

    for iv in intervals:
        ov = _overlap(seg.start, seg.end, iv.start, iv.end)
        if ov > 0:
            raw[iv.agent][0] += ov
            raw[iv.agent][1] += ov * iv.avg_confidence

    result: Dict[str, Tuple[float, float]] = {}
    for agent, (ov, wconf) in raw.items():
        result[agent] = (ov, wconf / ov if ov > 0 else 0.0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — hard matching
# ─────────────────────────────────────────────────────────────────────────────

def _build_speaker_agent_evidence(
    segments: List[ASRSegment],
    intervals: List[SpeakingInterval],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, SpeakerAgentEvidence]], Set[str]]:
    """Build the speaker × agent evidence matrix (coverage + confidence).

    Returns
    -------
    speaker_durations:
        Total speech duration per speaker.
    evidence:
        speaker → agent → SpeakerAgentEvidence (with coverage pre-computed).
    all_agents:
        All agent names seen in CV intervals.
    """
    speaker_durations: Dict[str, float] = defaultdict(float)
    evidence: Dict[str, Dict[str, SpeakerAgentEvidence]] = defaultdict(dict)
    all_agents: Set[str] = {iv.agent for iv in intervals}

    for seg in segments:
        duration = seg.end - seg.start
        if duration <= 0:
            continue

        speaker = seg.speaker
        speaker_durations[speaker] += duration

        for agent, (overlap, avg_conf) in _score_agents_for_segment(seg, intervals).items():
            if agent not in evidence[speaker]:
                evidence[speaker][agent] = SpeakerAgentEvidence()
            cell = evidence[speaker][agent]
            cell.overlap += overlap
            cell.weighted_conf_sum += overlap * avg_conf

    # Finalise coverage and avg_confidence
    for speaker, agent_map in evidence.items():
        total_dur = speaker_durations.get(speaker, 0.0)
        for cell in agent_map.values():
            cell.avg_confidence = (
                cell.weighted_conf_sum / cell.overlap if cell.overlap > 0 else 0.0
            )
            cell.coverage = cell.overlap / total_dur if total_dur > 0 else 0.0

    return speaker_durations, evidence, all_agents


def _derive_forbidden_pairs(
    speaker_durations: Dict[str, float],
    evidence: Dict[str, Dict[str, SpeakerAgentEvidence]],
    all_agents: Set[str],
    settings: Settings,
) -> Set[Tuple[str, str]]:
    """Mark pairs where coverage is too low to ever be plausible."""
    forbidden: Set[Tuple[str, str]] = set()
    threshold = settings.fusion_forbidden_coverage_threshold

    for speaker in speaker_durations:
        for agent in all_agents:
            cell = evidence.get(speaker, {}).get(agent)
            if (cell.coverage if cell else 0.0) < threshold:
                forbidden.add((speaker, agent))

    return forbidden


def _build_hard_map(
    speaker_durations: Dict[str, float],
    evidence: Dict[str, Dict[str, SpeakerAgentEvidence]],
    all_agents: Set[str],
    settings: Settings,
) -> Tuple[Dict[str, str], Set[Tuple[str, str]]]:
    """Produce hard (discrete) speaker→agent mapping.

    A pair is hard when:
    1. coverage  >= fusion_coverage_threshold
    2. avg_conf  >= fusion_confidence_threshold
    3. No near-equal second candidate: Δcoverage >= fusion_hard_delta
    4. No other speaker has a higher coverage for the same agent
    5. The agent is not already locked to another speaker
    """
    hard_map: Dict[str, str] = {}
    forbidden_pairs = _derive_forbidden_pairs(
        speaker_durations, evidence, all_agents, settings
    )
    used_agents: Set[str] = set()

    for speaker, total_dur in speaker_durations.items():
        if total_dur <= 0:
            continue

        candidates = [
            (agent, cell.coverage, cell.avg_confidence)
            for agent, cell in evidence.get(speaker, {}).items()
            if cell.coverage >= settings.fusion_coverage_threshold
            and cell.avg_confidence >= settings.fusion_confidence_threshold
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        if not candidates:
            logger.info("M5 – Speaker %s: no strong candidates.", speaker)
            continue

        best_agent, best_cov, best_conf = candidates[0]
        second_cov = candidates[1][1] if len(candidates) > 1 else 0.0

        if (best_cov - second_cov) < settings.fusion_hard_delta:
            logger.info(
                "M5 – Speaker %s undecided: best=%s (%.2f) vs second (%.2f), delta too small.",
                speaker, best_agent, best_cov, second_cov,
            )
            continue

        # Check no other speaker explains this agent better
        conflict = any(
            other_map.get(best_agent, SpeakerAgentEvidence()).coverage > best_cov
            for spk, other_map in evidence.items()
            if spk != speaker
        )
        if conflict:
            logger.info(
                "M5 – Speaker %s → %s rejected: agent better explained by another speaker.",
                speaker, best_agent,
            )
            continue

        if best_agent in used_agents:
            logger.info(
                "M5 – Speaker %s → %s rejected: agent already locked.",
                speaker, best_agent,
            )
            continue

        hard_map[speaker] = best_agent
        used_agents.add(best_agent)
        logger.info(
            "M5 – HARD MAP: %s → %s (coverage=%.2f, conf=%.2f)",
            speaker, best_agent, best_cov, best_conf,
        )

    logger.info(
        "M5 – Phase 1 done: hard_map=%d speakers, undecided=%d, forbidden_pairs=%d",
        len(hard_map),
        len(speaker_durations) - len(hard_map),
        len(forbidden_pairs),
    )
    return hard_map, forbidden_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — per-segment soft fusion
# ─────────────────────────────────────────────────────────────────────────────

def _build_global_scores(
    evidence: Dict[str, Dict[str, SpeakerAgentEvidence]],
) -> Dict[str, Dict[str, float]]:
    """Extract coverage-based global scores from the evidence matrix.

    Returns
    -------
    dict[speaker_id, dict[agent, coverage_score (0..1)]]
    """
    return {
        speaker: {agent: cell.coverage for agent, cell in agent_map.items()}
        for speaker, agent_map in evidence.items()
    }


def _compute_diar_score(
    seg: ASRSegment,
    agent: str,
    global_scores: Dict[str, Dict[str, float]],
    hard_map: Dict[str, str],
    hard_prior: float,
) -> float:
    """Compute the diarization-side score for *agent* given *seg*.

    Sources (in priority order):
    1. hard_map prior — if seg.speaker maps to this agent, return hard_prior.
    2. Weighted combination of global_scores for the primary speaker and
       any ASRSpeakerCandidates from M3.
    """
    # Hard map prior takes precedence
    if hard_map.get(seg.speaker) == agent:
        return hard_prior

    # Build candidate weights: {speaker_id: weight}
    weights: Dict[str, float] = {seg.speaker: 1.0}
    for c in seg.speaker_candidates:
        if c.speaker_id != seg.speaker:
            # keep the max weight if the same id appears twice
            weights[c.speaker_id] = max(weights.get(c.speaker_id, 0.0), c.weight)

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0

    weighted_sum = sum(
        w * global_scores.get(spk_id, {}).get(agent, 0.0)
        for spk_id, w in weights.items()
    )
    return weighted_sum / total_weight


def _compute_cv_score(
    seg: ASRSegment,
    agent: str,
    intervals: List[SpeakingInterval],
) -> float:
    """Compute the per-segment CV overlap score for *agent*.

    Score = (total overlap / segment duration) × avg_confidence_on_overlap.
    Clamped to [0, 1].
    """
    seg_duration = seg.end - seg.start
    if seg_duration <= 0:
        return 0.0

    total_overlap = 0.0
    weighted_conf = 0.0

    for iv in intervals:
        if iv.agent != agent:
            continue
        ov = _overlap(seg.start, seg.end, iv.start, iv.end)
        if ov > 0:
            total_overlap += ov
            weighted_conf += ov * iv.avg_confidence

    if total_overlap <= 0:
        return 0.0

    norm_overlap = min(1.0, total_overlap / seg_duration)
    avg_conf = weighted_conf / total_overlap
    return norm_overlap * avg_conf


def _fuse_segment_phase2(
        seg: ASRSegment,
        intervals: List[SpeakingInterval],
        hard_map: Dict[str, str],
        final_map: Dict[str, str],
        counts_match: bool,
        global_scores: Dict[str, Dict[str, float]],
        all_agents: Set[str],
        settings: Settings,
) -> FusedSegment:
    """Apply per-segment fusion combining M3 diar scores and M4 CV overlap."""

    alpha = settings.fusion_alpha
    beta = settings.fusion_beta
    hard_prior = settings.fusion_hard_prior
    winner_threshold = settings.fusion_winner_threshold

    candidates_pool: Set[str] = set(all_agents)
    for spk_id in [seg.speaker] + [c.speaker_id for c in seg.speaker_candidates]:
        candidates_pool.update(global_scores.get(spk_id, {}).keys())

    if seg.speaker in hard_map:
        candidates_pool.add(hard_map[seg.speaker])

    if counts_match and seg.speaker in final_map:
        candidates_pool.add(final_map[seg.speaker])

    scored: List[SpeakerAgentCandidate] = []
    for agent in candidates_pool:
        diar_s = _compute_diar_score(seg, agent, global_scores, hard_map, hard_prior)
        cv_s = _compute_cv_score(seg, agent, intervals)
        combined = alpha * diar_s + beta * cv_s
        scored.append(
            SpeakerAgentCandidate(
                agent=agent,
                score=round(combined, 6),
                diar_score=round(diar_s, 6),
                cv_score=round(cv_s, 6),
            )
        )

    scored.sort(key=lambda c: c.score, reverse=True)

    winner: Optional[SpeakerAgentCandidate] = None
    if scored and scored[0].score >= winner_threshold:
        winner = scored[0]

    resolved_speaker = _apply_final_map_to_segment(
        seg=seg,
        winner=winner,
        final_map=final_map,
        counts_match=counts_match,
        winner_threshold=winner_threshold,
    )

    if winner is None:
        confidence = 0.0
        source = "unresolved" if not (counts_match and seg.speaker in final_map) else "complete_map"
    else:
        runner_up_score = scored[1].score if len(scored) > 1 else 0.0
        confidence = round(winner.score - runner_up_score, 6)

        if counts_match and seg.speaker in final_map:
            if winner.agent == final_map[seg.speaker]:
                source = "complete_map+cv+diar" if winner.cv_score > 0 and winner.diar_score > 0 else "complete_map"
            else:
                source = "complete_map_override"
        elif winner.cv_score > 0 and winner.diar_score > 0:
            source = "cv+diar"
        elif winner.cv_score > 0:
            source = "cv_only"
        elif winner.diar_score > 0:
            source = "map_only"
        else:
            source = "unresolved"

    return FusedSegment(
        start=seg.start,
        end=seg.end,
        text=seg.text,
        speaker=resolved_speaker,
        speaker_candidates=scored,
        attribution_confidence=confidence,
        attribution_source=source,
    )


def _log_phase2_stats(fused: List[FusedSegment], settings: Settings) -> None:
    """Log attribution source distribution and low-confidence examples."""
    source_counts: Dict[str, int] = defaultdict(int)
    for fs in fused:
        source_counts[fs.attribution_source] += 1

    total = len(fused)
    logger.info(
        "M5 – Phase 2 attribution sources: cv+diar=%d, cv_only=%d, map_only=%d, "
        "unresolved=%d  (total=%d)",
        source_counts.get("cv+diar", 0),
        source_counts.get("cv_only", 0),
        source_counts.get("map_only", 0),
        source_counts.get("unresolved", 0),
        total,
    )

    # Log up to 5 most contested (low confidence) segments at DEBUG
    LOW_CONF_THRESHOLD = 0.10
    contested = [
        fs for fs in fused
        if fs.attribution_source != "unresolved"
        and fs.attribution_confidence < LOW_CONF_THRESHOLD
    ]
    if contested:
        logger.debug(
            "M5 – %d low-confidence segments (conf < %.2f):",
            len(contested), LOW_CONF_THRESHOLD,
        )
        for fs in contested[:5]:
            top2 = [(c.agent, c.score) for c in fs.speaker_candidates[:2]]
            logger.debug(
                "  [%.2f–%.2f] %r → %s (conf=%.3f, src=%s) top2=%s",
                fs.start, fs.end,
                fs.text[:40],
                fs.speaker,
                fs.attribution_confidence,
                fs.attribution_source,
                top2,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fuse(
        asr_segments: List[ASRSegment],
        cv_detections: List[CVDetection],
        settings: Settings,
        merge_gap: float = DEFAULT_MERGE_GAP,
) -> List[FusedSegment]:
    """Fuse ASR and CV data into speaker-labelled transcript segments."""
    if not cv_detections:
        logger.warning("M5 – no CV detections; all segments stay unresolved.")
        return [
            FusedSegment(
                start=s.start,
                end=s.end,
                speaker=s.speaker,
                text=s.text,
                attribution_source="unresolved",
            )
            for s in asr_segments
        ]

    intervals = _build_intervals(cv_detections, merge_gap)

    speaker_durations, evidence, all_agents = _build_speaker_agent_evidence(
        asr_segments, intervals
    )

    hard_map, forbidden_pairs = _build_hard_map(
        speaker_durations, evidence, all_agents, settings
    )

    unique_speakers = sorted(speaker_durations.keys())
    unique_agents = sorted(all_agents)
    counts_match = _should_force_complete_mapping(speaker_durations, all_agents)

    logger.info(
        "M5 – unique speakers=%d, unique agents=%d before complete mapping",
        len(unique_speakers),
        len(unique_agents),
    )
    logger.info("M5 – hard_map: %s", hard_map)
    logger.info("M5 – forbidden_pairs count: %d", len(forbidden_pairs))

    if counts_match:
        logger.info(
            "M5 – forcing complete speaker↔agent mapping because counts match: %d speakers, %d agents",
            len(unique_speakers),
            len(unique_agents),
        )
        final_map, fallback_assignments = _build_complete_map_when_counts_match(
            speaker_durations=speaker_durations,
            evidence=evidence,
            all_agents=all_agents,
            hard_map=hard_map,
        )
        logger.info("M5 – final_map: %s", final_map)

        if fallback_assignments:
            logger.warning(
                "M5 – complete mapping used %d fallback assignments with low/zero evidence: %s",
                len(fallback_assignments),
                fallback_assignments,
            )
    else:
        final_map = dict(hard_map)
        logger.info(
            "M5 – complete mapping disabled because counts differ: %d speakers vs %d agents",
            len(unique_speakers),
            len(unique_agents),
        )
        logger.info("M5 – final_map (same as hard_map in non-complete mode): %s", final_map)

    global_scores = _build_global_scores(evidence)

    fused: List[FusedSegment] = [
        _fuse_segment_phase2(
            seg=seg,
            intervals=intervals,
            hard_map=hard_map,
            final_map=final_map,
            counts_match=counts_match,
            global_scores=global_scores,
            all_agents=all_agents,
            settings=settings,
        )
        for seg in asr_segments
    ]

    _log_phase2_stats(fused, settings)

    unique_output_labels = sorted({fs.speaker for fs in fused})
    logger.info(
        "M5 – final unique output labels (%d): %s",
        len(unique_output_labels),
        unique_output_labels,
    )

    return fused
