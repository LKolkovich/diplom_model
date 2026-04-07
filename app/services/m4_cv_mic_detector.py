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
    match_template,
    multi_scale_match,
)

logger = logging.getLogger(__name__)

MIC_ICON_PATH = Path("templates") / "mic_active.png"

PORTRAIT_OFFSET_X = 2
PORTRAIT_OFFSET_Y = -2
PORTRAIT_W = 36
PORTRAIT_H = 36

MATCH_THRESHOLD = 0.60


def _load_mic_template() -> Optional[np.ndarray]:
    if MIC_ICON_PATH.exists():
        img = cv2.imread(str(MIC_ICON_PATH))
        if img is not None:
            logger.info("M4 – loaded mic-active template from %s", MIC_ICON_PATH)
            return img
    logger.warning(
        "M4 – mic-active template not found at '%s'. "
        "Falling back to brightness-based detection.",
        MIC_ICON_PATH,
    )
    return None


def _brightness_has_activity(roi_frame: np.ndarray, threshold: int = 200) -> bool:
    """Fallback heuristic: check for very bright pixels in the ROI.

    The Valorant speaking indicator lights up brightly.  When no mic template
    is available we use this as a coarse activity detector.
    """
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    ratio = np.count_nonzero(bright) / bright.size
    return bool(ratio > 0.02)


def _extract_portrait(
    roi_frame: np.ndarray,
    icon_top_left: tuple[int, int],
) -> np.ndarray:
    x = max(0, icon_top_left[0] + PORTRAIT_OFFSET_X)
    y = max(0, icon_top_left[1] + PORTRAIT_OFFSET_Y)
    x2 = min(roi_frame.shape[1], x + PORTRAIT_W)
    y2 = min(roi_frame.shape[0], y + PORTRAIT_H)
    return roi_frame[y:y2, x:x2]


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
    mic_template: Optional[np.ndarray] = _load_mic_template()

    detections: List[CVDetection] = []

    for ef in iter_frames(video_path, roi, target_fps):
        frame = ef.frame
        if frame.size == 0:
            continue

        icon_location: Optional[tuple[int, int]] = None

        if mic_template is not None:
            match = multi_scale_match(frame, mic_template, threshold=MATCH_THRESHOLD)
            if match:
                icon_location = match.top_left
        else:
            if not _brightness_has_activity(frame):
                continue
            icon_location = (0, 0)

        if icon_location is None:
            continue

        if agent_templates:
            portrait = _extract_portrait(frame, icon_location)
            agent_name, confidence = identify_agent(
                portrait, agent_templates, phash_threshold
            )
        else:
            agent_name = "unknown"
            confidence = 0.5

        if agent_name:
            detections.append(
                CVDetection(
                    frame_index=ef.index,
                    timestamp=ef.timestamp,
                    agent=agent_name,
                    confidence=confidence,
                )
            )
            logger.debug(
                "M4 – t=%.2fs agent=%s conf=%.2f", ef.timestamp, agent_name, confidence
            )

    logger.info("M4 – detected %d speaking events", len(detections))
    return detections
