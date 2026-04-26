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


def run_discovery_phase(
    video_path: str | Path,
    roi: Optional[tuple[float, float, float, float]],
    agent_templates: List[AgentTemplate],
    settings: Settings,
) -> tuple[List[AgentTemplate], float, Optional[tuple[int, int, int, int]]]:
    """
    Discovery Phase: Scans the video at a low FPS to identify active agents,
    optimal scale, and Anchor Zone.
    """
    logger.info("M4 – starting Discovery Phase (%.1f FPS full-video scan)...", settings.cv_discovery_fps)
    
    primary_scales = np.linspace(0.03, 0.15, num=15)
    fallback_scales = np.linspace(0.03, 0.30, num=15)
    
    # agent_name -> list of (scale, x, y, w, h)
    detections_by_agent: Dict[str, List[Tuple[float, int, int, int, int]]] = {}
    
    if not agent_templates:
        return [], 0.1, None

    template_size = agent_templates[0].image.shape[0]
    
    for ef in iter_frames(video_path, roi, target_fps=settings.cv_discovery_fps, debug_frames=False):
        frame = ef.frame
        if frame.size == 0:
            continue
            
        found_in_frame = False
        
        # Try primary scales first, then fallback
        for scales in [primary_scales, fallback_scales]:
            for scale in scales:
                new_size = max(1, int(template_size * scale))
                resized_templates = [
                    cv2.resize(at.image, (new_size, new_size), interpolation=cv2.INTER_AREA)
                    for at in agent_templates
                ]
                
                matches = match_templates(frame, resized_templates, threshold=settings.cv_discovery_threshold)
                if matches:
                    matches = non_max_suppression(matches)
                    for x, y, w, h, score, t_idx in matches:
                        agent_name = agent_templates[t_idx].name
                        if agent_name not in detections_by_agent:
                            detections_by_agent[agent_name] = []
                        detections_by_agent[agent_name].append((scale, x, y, w, h))
                        found_in_frame = True
            
            if found_in_frame:
                break
    
    # Filter agents by min detections
    active_agents_info = {
        name: detections
        for name, detections in detections_by_agent.items()
        if len(detections) >= settings.cv_discovery_min_detections
    }
    
    if not active_agents_info:
        logger.warning("M4 – Discovery Phase FAILED: no agents found with >= %d detections", 
                       settings.cv_discovery_min_detections)
        return agent_templates, 0.1, None
    
    active_agent_names = sorted(list(active_agents_info.keys()))
    active_templates = [at for at in agent_templates if at.name in active_agent_names]
    
    # Calculate optimal scale (median of all detections)
    all_scales = [d[0] for detections in active_agents_info.values() for d in detections]
    optimal_scale = float(np.median(all_scales))
    
    # Calculate Anchor Zone
    all_x = [d[1] for detections in active_agents_info.values() for d in detections]
    all_y = [d[2] for detections in active_agents_info.values() for d in detections]
    all_w = [d[3] for detections in active_agents_info.values() for d in detections]
    all_h = [d[4] for detections in active_agents_info.values() for d in detections]
    
    min_x = min(all_x)
    min_y = min(all_y)
    max_x = max([x + w for x, w in zip(all_x, all_w)])
    max_y = max([y + h for y, h in zip(all_y, all_h)])
    
    # Formula: (min-5/max+50)
    anchor_zone = (
        max(0, int(min_x - 5)),
        max(0, int(min_y - 5)),
        int(max_x + 50),
        int(max_y + 50)
    )
    
    logger.info("M4 – Discovery Phase COMPLETE: active agents: %s, optimal scale: %.3f, Anchor Zone: %s",
                active_agent_names, optimal_scale, anchor_zone)
    
    return active_templates, optimal_scale, anchor_zone


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
    # Support CV_AGENTS_LIST
    agents_list = None
    if settings.cv_agents_list:
        agents_list = [a.strip() for a in settings.cv_agents_list.split(",")]
    
    agent_templates: List[AgentTemplate] = load_agent_templates(agent_templates_dir, agents_list=agents_list)
    
    if not agent_templates:
        logger.error("No agent templates found in %s", agent_templates_dir)
        raise ValueError(f"No agent templates found in {agent_templates_dir}")

    # 1. Discovery Phase
    if settings.cv_skip_discovery:
        logger.info("M4 – skipping Discovery Phase as per settings")
        active_templates = agent_templates
        # If discovery skipped, use a fallback scale
        # Trying to find a reasonable default scale if we don't have one
        # Current logic used cv_fallback_size_percent, let's keep a similar fallback
        optimal_scale = 0.1
        anchor_zone = None
    else:
        active_templates, optimal_scale, anchor_zone = run_discovery_phase(
            video_path=video_path,
            roi=roi,
            agent_templates=agent_templates,
            settings=settings,
        )
    
    # Prepare resized images for the active agents
    template_size = active_templates[0].image.shape[0]
    new_size = max(1, int(template_size * optimal_scale))
    agent_images = [
        cv2.resize(at.image, (new_size, new_size), interpolation=cv2.INTER_AREA)
        for at in active_templates
    ]
    
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
        video_path, roi, target_fps, debug_frames=False
    ):
        ts = ef.timestamp
        frame = ef.frame
        if frame.size == 0:
            continue

        total_frames += 1
        
        # Crop to Anchor Zone if available
        search_frame = frame
        offset_x, offset_y = 0, 0
        if anchor_zone:
            x1, y1, x2, y2 = anchor_zone
            # Clip to frame boundaries
            x1 = max(0, min(x1, frame.shape[1] - 1))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            x2 = max(x1 + 1, min(x2, frame.shape[1]))
            y2 = max(y1 + 1, min(y2, frame.shape[0]))
            search_frame = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1

        # Detect portraits
        portrait_boxes = match_templates(search_frame, agent_images, threshold=settings.cv_portrait_threshold)
        portrait_boxes = non_max_suppression(portrait_boxes)
        
        # Translate coordinates back to ROI-relative
        if anchor_zone:
            translated_boxes = []
            for x, y, w_b, h_b, s, p_idx in portrait_boxes:
                translated_boxes.append((x + offset_x, y + offset_y, w_b, h_b, s, p_idx))
            portrait_boxes = translated_boxes
        
        active_this_frame: Dict[str, float] = {}
        
        for pb in portrait_boxes:
            px, py, pw, ph, ps, p_idx = pb
            agent_name = active_templates[p_idx].name
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
                # Draw Anchor Zone if it exists
                if anchor_zone:
                    x1, y1, x2, y2 = anchor_zone
                    # Clip for drawing
                    x1 = max(0, min(x1, frame.shape[1] - 1))
                    y1 = max(0, min(y1, frame.shape[0] - 1))
                    x2 = max(x1 + 1, min(x2, frame.shape[1]))
                    y2 = max(y1 + 1, min(y2, frame.shape[0]))
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

                for pb in portrait_boxes:
                    x, y, w_b, h_b, s, p_idx in pb
                    cv2.rectangle(debug_img, (x, y), (x + w_b, y + h_b), (255, 0, 0), 2)
                    label = f"{active_templates[p_idx].name} ({s:.2f})"
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
    
    all_agent_names = {at.name for at in active_templates}
    
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
