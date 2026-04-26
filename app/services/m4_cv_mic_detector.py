"""M4 – CV Mic Detector.

Analyses ROI frames from M2 to detect which Valorant agent is speaking at
each point in time.

Format cv_detections.jsonl (one JSON per line):
{
  "timestamp_ms": 12345,           // absolute timestamp of the frame in milliseconds
  "frame_index": 42,              // frame index within the M2 iteration
  "detections": {                 // detections per agent in this frame
    "jett": 0.91,
    "sage": 0.78
  }
}

- timestamp_ms: used for manual validation (sync with video).
- detections: confidence score for each agent in this frame after portrait detection.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import List, Optional, Dict

import cv2
# pylint: disable=unused-import
import numpy as np

from app.models import CVDetection
from app.services.m2_frame_extractor import iter_frames
from app.utils.cv_logic import (
    AgentTemplate,
    load_agent_templates,
    match_templates,
    non_max_suppression,
)
from app.config import Settings

logger = logging.getLogger(__name__)


def _get_fallback_scale(frame_height: int, ref_template_height: int, settings: Settings) -> float:
    """Calculate fallback scale based on 8% of ROI height."""
    target_height = frame_height * settings.cv_fallback_size_percent
    return target_height / ref_template_height


def calibrate_templates(
    video_path: str | Path,
    roi: Optional[tuple[float, float, float, float]],
    target_fps: int,
    agent_templates: List[AgentTemplate],
    settings: Settings,
) -> float:
    """Two-stage automatic template calibration."""
    if not agent_templates:
        return 1.0

    ref_template = agent_templates[0].image
    ref_h, _ = ref_template.shape[:2]

    coarse_frames = []
    all_calib_frames = []
    for i, ef in enumerate(iter_frames(video_path, roi, target_fps)):
        if i >= settings.cv_calibration_max_frames:
            break
        all_calib_frames.append(ef.frame)
        if i % 5 == 0:
            coarse_frames.append(ef.frame)

    if not all_calib_frames:
        logger.warning("No frames found for calibration, using default scale 1.0")
        return 1.0

    # Stage 1: Coarse
    coarse_scales = [round(0.5 + i * 0.1, 1) for i in range(11)]  # 0.5 to 1.5
    scale_scores = []

    for scale in coarse_scales:
        resized = cv2.resize(ref_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        max_score = 0.0
        for frame in coarse_frames:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray_resized.shape[0] > gray_frame.shape[0] or gray_resized.shape[1] > gray_frame.shape[1]:
                continue
            res = cv2.matchTemplate(gray_frame, gray_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            max_score = max(max_score, max_val)
        scale_scores.append((scale, max_score))

    scale_scores.sort(key=lambda x: x[1], reverse=True)
    top2 = scale_scores[:2]

    if top2[0][1] < settings.cv_portrait_threshold:
        fallback = _get_fallback_scale(all_calib_frames[0].shape[0], ref_h, settings)
        logger.info(
            "Calibration FALLBACK: max score %.2f < threshold %.2f. Using fallback scale %.2f",
            top2[0][1],
            settings.cv_portrait_threshold,
            fallback,
        )
        return fallback

    # Stage 2: Fine
    h1 = int(ref_h * top2[0][0])
    h2 = int(ref_h * top2[1][0])
    min_h = max(1, min(h1, h2) - 2)
    max_h = max(min_h + 1, max(h1, h2) + 2)

    fine_scores = []
    for h in range(min_h, max_h + 1):
        scale = h / ref_h
        resized = cv2.resize(ref_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        max_score = 0.0
        for frame in all_calib_frames:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray_resized.shape[0] > gray_frame.shape[0] or gray_resized.shape[1] > gray_frame.shape[1]:
                continue
            res = cv2.matchTemplate(gray_frame, gray_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            max_score = max(max_score, max_val)
        fine_scores.append((scale, max_score))

    fine_scores.sort(key=lambda x: x[1], reverse=True)
    best_scale, best_score = fine_scores[0]

    logger.info("Calibration SUCCESS: scale %.2f (score %.2f)", best_scale, best_score)
    return best_scale


def detect_speakers(
    video_path: str | Path,
    roi: Optional[tuple[float, float, float, float]],
    target_fps: int,
    agent_templates_dir: str | Path,
    settings: Settings,
    debug_frames: bool = False,
    debug_frames_dir: Path | str = "debug_frames",
) -> tuple[List[CVDetection], int]:
    """Run the full CV portrait-detection pass over the video."""
    agent_templates: List[AgentTemplate] = load_agent_templates(agent_templates_dir)
    
    if not agent_templates:
        logger.warning("No agent templates found in %s", agent_templates_dir)
        return [], 0

    # 1. Calibrate templates
    calibrated_scale = calibrate_templates(video_path, roi, target_fps, agent_templates, settings)
    
    # 2. Resize agent images
    agent_images = []
    for at in agent_templates:
        resized = cv2.resize(at.image, None, fx=calibrated_scale, fy=calibrated_scale, interpolation=cv2.INTER_AREA)
        agent_images.append(resized)
    
    raw_frame_detections: List[Dict[str, float]] = [] # list of {agent_name: confidence}
    timestamps: List[float] = []
    
    total_frames: int = 0

    if debug_frames:
        debug_dir = Path(debug_frames_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv_debug_log_path = debug_dir / "cv_detections.jsonl"
        cv_debug_log = open(cv_debug_log_path, "w")
    else:
        cv_debug_log = None

    for ef in iter_frames(
        video_path, roi, target_fps, debug_frames=False # Don't let M2 save raw debug frames, M4 will do it with overlays
    ):
        ts = ef.timestamp
        frame = ef.frame
        if frame.size == 0:
            continue

        total_frames += 1
        
        # Detect portraits
        portrait_boxes = match_templates(frame, agent_images, threshold=settings.cv_portrait_threshold)
        portrait_boxes = non_max_suppression(portrait_boxes)
        
        active_this_frame: Dict[str, float] = {}
        
        for pb in portrait_boxes:
            px, py, pw, ph, ps, p_idx = pb
            agent_name = agent_templates[p_idx].name
            # Use max confidence if same agent detected multiple times (shouldn't happen with NMS)
            active_this_frame[agent_name] = max(active_this_frame.get(agent_name, 0.0), ps)

        raw_frame_detections.append(active_this_frame)
        timestamps.append(ts)

        if cv_debug_log:
            log_entry = {
                "timestamp_ms": int(ts * 1000),
                "frame_index": ef.index,
                "detections": active_this_frame
            }
            cv_debug_log.write(json.dumps(log_entry) + "\n")
            
            if debug_frames:
                # Draw overlays
                debug_img = frame.copy()
                for pb in portrait_boxes:
                    x, y, w_b, h_b, s, p_idx = pb
                    cv2.rectangle(debug_img, (x, y), (x + w_b, y + h_b), (255, 0, 0), 2)
                    label = f"{agent_templates[p_idx].name} ({s:.2f})"
                    cv2.putText(debug_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                ts_ms = int(ts * 1000)
                dest = Path(debug_frames_dir) / f"frame_{ef.index:06d}_{ts_ms:08d}.png"
                cv2.imwrite(str(dest), debug_img)

    if cv_debug_log:
        cv_debug_log.close()

    # 4. Temporal smoothing (K-of-N)
    k = settings.cv_temporal_k
    n = settings.cv_temporal_n
    
    smoothed_detections: List[CVDetection] = []
    
    all_agent_names = {at.name for at in agent_templates}
    
    for agent_name in all_agent_names:
        history = [0] * len(raw_frame_detections)
        for i, detections in enumerate(raw_frame_detections):
            if agent_name in detections:
                history[i] = 1
        
        # Sliding window
        for i in range(len(history)):
            window = history[max(0, i - n + 1) : i + 1]
            if sum(window) >= k:
                # Agent is active at this timestamp
                conf = raw_frame_detections[i].get(agent_name, 0.0)
                # If they are active due to smoothing but not detected in current frame, 
                # we need a confidence. Let's use the average of detections in window?
                if conf == 0.0:
                    detected_confs = [raw_frame_detections[j][agent_name] for j in range(max(0, i-n+1), i+1) if agent_name in raw_frame_detections[j]]
                    conf = sum(detected_confs) / len(detected_confs) if detected_confs else 0.5
                
                smoothed_detections.append(
                    CVDetection(timestamp=timestamps[i], agent_name=agent_name, confidence=conf)
                )

    # Sort detections by timestamp
    smoothed_detections.sort(key=lambda x: x.timestamp)

    logger.info("M4 – detected %d speaking events across %d frames (after smoothing)", len(smoothed_detections), total_frames)
    return smoothed_detections, total_frames
