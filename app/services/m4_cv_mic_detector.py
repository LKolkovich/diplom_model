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
from typing import List, Optional, Dict, Tuple

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


def calibrate_templates(
    video_path: str | Path,
    roi: Optional[tuple[float, float, float, float]],
    target_fps: int,
    agent_templates: List[AgentTemplate],
    settings: Settings,
    max_calibration_frames: int = 30,
) -> tuple[list[np.ndarray], int]:
    """Two-stage automatic template calibration."""
    if not agent_templates:
        raise ValueError("No agent templates provided for calibration")

    logger.info("M4 – starting template calibration (two-stage)...")

    # We use the first template as a reference for sizing
    ref_at = agent_templates[0]
    original_size = ref_at.image.shape[0] # Assuming square templates as per patterns

    # Stage 1: Coarse search
    logger.info("M4 – Stage 1: coarse search (15 scales, max %d frames)", max_calibration_frames)
    scales_coarse = np.linspace(0.1, 1.5, num=15)
    
    top_matches: List[Tuple[float, float, int, np.ndarray, int]] = [] # (scale, conf, frame_idx, frame_copy, agent_idx)

    found_any = False
    frame_count = 0
    for ef in iter_frames(video_path, roi, target_fps, debug_frames=False):
        if frame_count >= max_calibration_frames:
            break
        
        frame = ef.frame
        if frame.size == 0:
            continue
        
        frame_count += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for scale in scales_coarse:
            new_size = max(1, int(original_size * scale))
            
            for at_idx, at in enumerate(agent_templates):
                resized = cv2.resize(at.image, (new_size, new_size), interpolation=cv2.INTER_AREA)
                gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                
                if gray_resized.shape[0] > gray_frame.shape[0] or gray_resized.shape[1] > gray_frame.shape[1]:
                    continue
                
                res = cv2.matchTemplate(gray_frame, gray_resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                
                if max_val >= settings.cv_portrait_threshold:
                    top_matches.append((float(scale), float(max_val), ef.index, frame.copy(), at_idx))
                    found_any = True
        
        if found_any:
            logger.info("M4 – found %d matches on frame #%d", len(top_matches), ef.index)
            break

    if not top_matches:
        logger.warning("M4 – coarse search timeout after %d frames", frame_count)
        logger.warning("M4 – coarse search FAILED: no agents detected in calibration window")
        
        # Fallback logic
        # Need a frame to get height. If we have no frames at all, we might be in trouble but iter_frames should have yielded something if video is valid.
        # Let's try to get one frame if we haven't already.
        if frame_count == 0:
             # This should ideally not happen if the video is valid and has frames
             logger.error("M4 – no frames available even for fallback")
             fallback_size_px = max(1, int(100 * settings.cv_fallback_size_percent)) # Totally arbitrary if no frame
             frame_h = 100
        else:
             # We use the last frame seen
             frame_h = frame.shape[0]
             fallback_size_px = max(1, int(frame_h * settings.cv_fallback_size_percent))
        
        logger.warning("M4 – using FALLBACK: %dpx (%.0f%% of ROI height %dpx)", 
                       fallback_size_px, settings.cv_fallback_size_percent * 100, frame_h)
        
        resized_templates = [
            cv2.resize(at.image, (fallback_size_px, fallback_size_px), interpolation=cv2.INTER_AREA)
            for at in agent_templates
        ]
        return resized_templates, fallback_size_px

    # Stage 2: Fine search
    top_matches.sort(key=lambda x: x[1], reverse=True)
    top_2_scales = [
        top_matches[0][0],
        top_matches[1][0] if len(top_matches) > 1 else top_matches[0][0],
    ]
    
    logger.info("M4 – Stage 1 COMPLETE: top scales [%.2f, %.2f], confidences [%.3f, %.3f]",
                top_2_scales[0], top_2_scales[1], 
                top_matches[0][1], top_matches[1][1] if len(top_matches) > 1 else top_matches[0][1])

    min_size_px = max(1, int(original_size * min(top_2_scales)))
    max_size_px = max(1, int(original_size * max(top_2_scales)))

    calibration_frame = top_matches[0][3]
    best_agent_idx = top_matches[0][4]
    ref_at_fine = agent_templates[best_agent_idx]
    gray_calib_frame = cv2.cvtColor(calibration_frame, cv2.COLOR_BGR2GRAY)
    
    logger.info("M4 – Stage 2: fine search (%d sizes from %dpx to %dpx, step 1px)", 
                max_size_px - min_size_px + 1, min_size_px, max_size_px)
    
    best_size_px = min_size_px
    best_confidence = -1.0
    
    for size_px in range(min_size_px, max_size_px + 1):
        resized = cv2.resize(ref_at_fine.image, (size_px, size_px), interpolation=cv2.INTER_AREA)
        gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        if gray_resized.shape[0] > gray_calib_frame.shape[0] or gray_resized.shape[1] > gray_calib_frame.shape[1]:
            continue
            
        res = cv2.matchTemplate(gray_calib_frame, gray_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val > best_confidence:
            best_confidence = max_val
            best_size_px = size_px

    logger.info("M4 – Stage 2 COMPLETE: optimal size=%dpx, confidence=%.3f", best_size_px, best_confidence)
    logger.info("M4 – calibration SUCCESS: all %d templates %dx%d → %dx%d px", 
                len(agent_templates), original_size, original_size, best_size_px, best_size_px)

    resized_templates = [
        cv2.resize(at.image, (best_size_px, best_size_px), interpolation=cv2.INTER_AREA)
        for at in agent_templates
    ]
    return resized_templates, best_size_px


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
        logger.error("No agent templates found in %s", agent_templates_dir)
        # Match detailed requirements: guard before calling it and fail clearly
        raise ValueError(f"No agent templates found in {agent_templates_dir}")

    # 1. Calibrate templates
    agent_images, calibrated_size = calibrate_templates(
        video_path=video_path,
        roi=roi,
        target_fps=target_fps,
        agent_templates=agent_templates,
        settings=settings,
        max_calibration_frames=settings.cv_calibration_max_frames,
    )
    
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
