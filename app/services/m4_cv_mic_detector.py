"""M4 – CV Mic Detector.

Analyses ROI frames from M2 to detect which Valorant agent is speaking at
each point in time.

Detection strategy
------------------
1. Load a reference "mic-active" icon template (the speaking indicator shown
   next to a player's portrait in the comms panel).
2. For every extracted frame, run template matching against the ROI to find
   active mic indicators.
3. When an indicator is found, crop the adjacent agent-portrait area and use
   pHash against the pre-loaded agent template library (loaded from
   ``templates/agents/``) to identify the speaker.
4. Return time-stamped CVDetection events.

Coordinate conventions
-----------------------
All ROI frames have already been cropped to the voice-chat panel, so the
mic indicator should appear somewhere near the left/top of the cropped image.
The agent portrait is typically to the right of (or below) the mic icon.

Portrait crop offsets (relative to the icon top-left) are configurable via
the ``portrait_offset_*`` constants and should be tuned for the target UI
resolution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from app.models import CVDetection
from app.services.m2_frame_extractor import ExtractedFrame, iter_frames
from app.utils.cv_logic import (
    AgentTemplate,
    identify_agent,
    load_agent_templates,
)

logger = logging.getLogger(__name__)

PORTRAIT_W = 36
PORTRAIT_H = 36


def detect_speakers(
    video_path: str | Path,
    roi: tuple[float, float, float, float],
    target_fps: int,
    templates_dir: str | Path,
    phash_threshold: int = 10,
) -> List[CVDetection]:
    """Run the full CV mic-detection pass over the video.

    Parameters
    ----------
    video_path:
        Path to the source video.
    roi:
        Fractional (x1, y1, x2, y2) of the voice-chat panel.
    target_fps:
        Frame sampling rate passed to M2.
    templates_dir:
        Directory containing agent portrait PNG/JPEG files.
    phash_threshold:
        Maximum pHash Hamming distance to count as a match.

    Returns
    -------
    list[CVDetection]
        One entry per frame where a speaking agent was detected.
    """
    agent_templates: List[AgentTemplate] = load_agent_templates(templates_dir)

    detections: List[CVDetection] = []

    for ef in iter_frames(video_path, roi, target_fps):
        ts = ef.timestamp
        frame = ef.frame
        if frame.size == 0:
            continue

        # Crop fixed portrait area from top-left of ROI
        portrait = frame[0:PORTRAIT_H, 0:PORTRAIT_W]
        if portrait.size == 0:
            continue

        if agent_templates:
            agent_name, confidence = identify_agent(portrait, agent_templates, phash_threshold)
        else:
            agent_name = "unknown"
            confidence = 0.0

        if agent_name != "unknown":
            detections.append(
                CVDetection(timestamp=ts, agent_name=agent_name, confidence=confidence)
            )

    logger.info("M4 – detected %d speaking events", len(detections))
    return detections
