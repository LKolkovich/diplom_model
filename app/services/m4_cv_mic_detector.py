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
- detections: confidence score for each agent in this frame after mic ↔ portrait linking.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set

import cv2
import numpy as np

from app.models import CVDetection
from app.services.m2_frame_extractor import ExtractedFrame, iter_frames
from app.utils.cv_logic import (
    AgentTemplate,
    identify_agent,
    load_agent_templates,
    load_simple_templates,
    match_templates,
    non_max_suppression,
)
from app.config import Settings

logger = logging.getLogger(__name__)


def detect_speakers(
    video_path: str | Path,
    roi: Optional[tuple[float, float, float, float]],
    target_fps: int,
    agent_templates_dir: str | Path,
    mic_templates_dir: str | Path,
    settings: Settings,
    debug_frames: bool = False,
    debug_frames_dir: Path | str = "debug_frames",
) -> tuple[List[CVDetection], int]:
    """Run the full CV mic-detection pass over the video."""
    agent_templates: List[AgentTemplate] = load_agent_templates(agent_templates_dir)
    mic_templates: List[np.ndarray] = load_simple_templates(mic_templates_dir)
    
    if not agent_templates:
        logger.warning("No agent templates found in %s", agent_templates_dir)
    if not mic_templates:
        logger.warning("No mic templates found in %s", mic_templates_dir)

    agent_images = [at.image for at in agent_templates]
    
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
        video_path, roi, target_fps, debug_raw_frames=False # Don't let M2 save raw debug frames, M4 will do it with overlays
    ):
        ts = ef.timestamp
        frame = ef.frame
        if frame.size == 0:
            continue

        total_frames += 1
        h, w = frame.shape[:2]
        
        # 1. Detect mic icons
        mic_boxes = match_templates(frame, mic_templates, threshold=settings.cv_mic_threshold)
        mic_boxes = non_max_suppression(mic_boxes)
        
        # 2. Detect portraits
        portrait_boxes = match_templates(frame, agent_images, threshold=settings.cv_portrait_threshold)
        portrait_boxes = non_max_suppression(portrait_boxes)
        
        active_this_frame: Dict[str, float] = {}
        linked_mic_to_portrait = [] # for debug drawing
        
        for mb in mic_boxes:
            mx, my, mw, mh, ms, _ = mb
            m_cx, m_cy = mx + mw/2, my + mh/2
            
            best_pb = None
            min_dist_sq = float('inf')
            
            for pb in portrait_boxes:
                px, py, pw, ph, ps, p_idx = pb
                p_cx, p_cy = px + pw/2, py + ph/2
                
                dx = abs(m_cx - p_cx) / w
                dy = abs(m_cy - p_cy) / h
                
                if dx <= settings.cv_dx_limit and dy <= settings.cv_dy_limit:
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        best_pb = pb
            
            if best_pb:
                px, py, pw, ph, ps, p_idx = best_pb
                agent_name = agent_templates[p_idx].name
                # Use max confidence if same agent detected multiple times (shouldn't happen with NMS)
                active_this_frame[agent_name] = max(active_this_frame.get(agent_name, 0.0), ps)
                linked_mic_to_portrait.append((mb, best_pb))

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
                for mb in mic_boxes:
                    x, y, w_b, h_b, s, _ = mb
                    cv2.rectangle(debug_img, (x, y), (x + w_b, y + h_b), (0, 255, 0), 2)
                for pb in portrait_boxes:
                    x, y, w_b, h_b, s, p_idx = pb
                    cv2.rectangle(debug_img, (x, y), (x + w_b, y + h_b), (255, 0, 0), 2)
                    cv2.putText(debug_img, agent_templates[p_idx].name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                for mb, pb in linked_mic_to_portrait:
                    m_cx, m_cy = int(mb[0] + mb[2]/2), int(mb[1] + mb[3]/2)
                    p_cx, p_cy = int(pb[0] + pb[2]/2), int(pb[1] + pb[3]/2)
                    cv2.line(debug_img, (m_cx, m_cy), (p_cx, p_cy), (0, 255, 255), 1)
                
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
