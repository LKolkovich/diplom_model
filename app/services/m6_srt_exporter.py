"""M6 – SRT Exporter.

Converts FusedSegment lists into SubRip (SRT) subtitle files:
- One combined SRT with speaker prefixes, e.g. ``[Jett]: Rush B!``.
- One SRT per agent containing only their lines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from app.models import FusedSegment

logger = logging.getLogger(__name__)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp ``HH:MM:SS,mmm``."""
    s = max(0.0, seconds)
    hours = int(s // 3600)
    minutes = int((s % 3600) // 60)
    secs = int(s % 60)
    millis = int(round((s - int(s)) * 1000))
    millis = min(millis, 999)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _build_srt(segments: List[FusedSegment], include_speaker: bool = False) -> str:
    """Render *segments* as SRT text."""
    lines: List[str] = []
    for idx, seg in enumerate(segments, start=1):
        text = f"[{seg.speaker}]: {seg.text}" if include_speaker else seg.text
        lines.append(str(idx))
        lines.append(f"{_format_timestamp(seg.start)} --> {_format_timestamp(seg.end)}")
        lines.append(text.strip())
        lines.append("")
    return "\n".join(lines)


def export(
    segments: List[FusedSegment],
    output_dir: str | Path,
    task_id: str,
) -> Dict[str, str]:
    """Write SRT files and return a mapping of label → file path.

    Parameters
    ----------
    segments:
        Fused, speaker-resolved segments from M5.
    output_dir:
        Directory to write the SRT files into.
    task_id:
        Used to namespace the output filenames.

    Returns
    -------
    dict
        Keys are ``"combined"`` and per-agent names; values are file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result: Dict[str, str] = {}

    combined_path = out / f"{task_id}_combined.srt"
    combined_path.write_text(_build_srt(segments, include_speaker=True), encoding="utf-8")
    result["combined"] = str(combined_path)
    logger.info("M6 – wrote combined SRT: %s", combined_path)

    by_speaker: Dict[str, List[FusedSegment]] = {}
    for seg in segments:
        by_speaker.setdefault(seg.speaker, []).append(seg)

    for speaker, spk_segs in sorted(by_speaker.items()):
        safe_name = speaker.replace(" ", "_").replace("/", "-")
        spk_path = out / f"{task_id}_{safe_name}.srt"
        spk_path.write_text(_build_srt(spk_segs, include_speaker=False), encoding="utf-8")
        result[speaker] = str(spk_path)
        logger.info("M6 – wrote speaker SRT: %s (%d segments)", spk_path, len(spk_segs))

    return result
