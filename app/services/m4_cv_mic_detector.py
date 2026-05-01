"""
M4 – CV Mic Detector.

Анализ ROI-кадров из M2, чтобы определить, какой агент Valorant говорит
в каждый момент времени.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import Settings
from app.models import CVDetection, DiscoveryResult
from app.services.m2_frame_extractor import iter_frames
from app.utils.cv_logic import (
    AgentTemplate,
    load_agent_templates,
    match_templates,
    non_max_suppression,
)

logger = logging.getLogger(__name__)


# -------------------- Preprocessing --------------------


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _normalize_blur_kernel(kernel: int) -> int:
    kernel = max(1, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def preprocess_for_template_matching(image: np.ndarray, settings: Settings) -> np.ndarray:
    """
    Конфигурируемая предобработка перед cv2.matchTemplate.
    Поддерживает:
    - none
    - gray
    - gray_clahe
    - gray_blur
    - gray_clahe_blur
    - edges (Canny)
    """

    mode = (settings.cv_preprocess_mode or "none").strip().lower()

    if mode == "none":
        return image

    gray = _ensure_gray(image)

    if mode == "gray":
        return gray

    if mode == "gray_clahe":
        clahe = cv2.createCLAHE(
            clipLimit=float(settings.cv_preprocess_clahe_clip_limit),
            tileGridSize=(
                int(settings.cv_preprocess_clahe_tile_size),
                int(settings.cv_preprocess_clahe_tile_size),
            ),
        )
        return clahe.apply(gray)

    if mode == "gray_blur":
        kernel = _normalize_blur_kernel(settings.cv_preprocess_blur_kernel)
        return cv2.GaussianBlur(gray, (kernel, kernel), 0)

    if mode == "gray_clahe_blur":
        clahe = cv2.createCLAHE(
            clipLimit=float(settings.cv_preprocess_clahe_clip_limit),
            tileGridSize=(
                int(settings.cv_preprocess_clahe_tile_size),
                int(settings.cv_preprocess_clahe_tile_size),
            ),
        )
        processed = clahe.apply(gray)
        kernel = _normalize_blur_kernel(settings.cv_preprocess_blur_kernel)
        return cv2.GaussianBlur(processed, (kernel, kernel), 0)

    if mode == "edges":
        return cv2.Canny(
            gray,
            threshold1=int(settings.cv_preprocess_canny_threshold1),
            threshold2=int(settings.cv_preprocess_canny_threshold2),
        )

    raise ValueError(f"Unsupported cv_preprocess_mode: {settings.cv_preprocess_mode}")


# -------------------- Геометрия стека агентов --------------------


def _cluster_is_valid(
        cluster: List[Tuple[int, int, int, int, float, int]],
        icon_size: int,
        settings: Settings,
) -> bool:
    """Проверяем, что кластер похож на вертикальную колонку иконок агентов."""

    if len(cluster) <= 1:
        return True

    boxes_sorted = sorted(cluster, key=lambda b: b[1])  # по верхней границе y

    centers_x = [(x + w / 2.0) for x, y, w, h, score, p_idx in boxes_sorted]
    if max(centers_x) - min(centers_x) > settings.cv_cluster_horizontal_tol * icon_size:
        return False

    for i in range(1, len(boxes_sorted)):
        prev = boxes_sorted[i - 1]
        cur = boxes_sorted[i]

        px, py, pw, ph, _, _ = prev
        cx, cy, cw, ch, _, _ = cur

        prev_center_x = px + pw / 2.0
        cur_center_x = cx + cw / 2.0
        dx = abs(cur_center_x - prev_center_x)
        if dx > settings.cv_cluster_horizontal_tol * icon_size:
            return False

        prev_center_y = py + ph / 2.0
        cur_center_y = cy + ch / 2.0
        center_delta = cur_center_y - prev_center_y

        agent_pos_delta = cy - (py + ph)

        center_ok = center_delta <= 2.0 * icon_size
        pos_ok = agent_pos_delta <= 1.0 * icon_size

        if not (center_ok or pos_ok):
            return False

        if 1.0 < agent_pos_delta / icon_size < 2.0:
            gap_ratio = agent_pos_delta / icon_size
            if abs(gap_ratio - 1.5) <= settings.cv_cluster_mid_gap_bias:
                return False

    return True

def _find_best_vertical_cluster(
        portrait_boxes: List[Tuple[int, int, int, int, float, int]],
        icon_size: int,
        settings: Settings,
) -> List[Tuple[int, int, int, int, float, int]]:
    """
    Выбираем лучший вертикальный стек агентов по геометрии и суммарному score.
    """

    if len(portrait_boxes) <= 1:
        return portrait_boxes

    boxes = sorted(portrait_boxes, key=lambda b: b[1] + b[3] / 2.0)
    best_cluster: List[Tuple[int, int, int, int, float, int]] = []
    best_score = -1.0

    for start_idx in range(len(boxes)):
        cluster = [boxes[start_idx]]
        last_box = boxes[start_idx]

        for next_idx in range(start_idx + 1, len(boxes)):
            candidate = boxes[next_idx]

            lx, ly, lw, lh, _, _ = last_box
            cx, cy, cw, ch, _, _ = candidate

            last_center_x = lx + lw / 2.0
            cand_center_x = cx + cw / 2.0
            dx = abs(cand_center_x - last_center_x)

            last_center_y = ly + lh / 2.0
            cand_center_y = cy + ch / 2.0
            center_delta = cand_center_y - last_center_y

            agent_pos_delta = cy - (ly + lh)

            horizontally_ok = dx <= settings.cv_cluster_horizontal_tol * icon_size
            vertically_ok = (
                    center_delta <= 2.0 * icon_size
                    or agent_pos_delta <= 1.0 * icon_size
            )

            suspicious_gap = False
            if icon_size > 0 and 1.0 < agent_pos_delta / icon_size < 2.0:
                gap_ratio = agent_pos_delta / icon_size
                suspicious_gap = abs(gap_ratio - 1.5) <= settings.cv_cluster_mid_gap_bias

            if horizontally_ok and vertically_ok and not suspicious_gap:
                existing_agents = {b[5] for b in cluster}
                if candidate[5] not in existing_agents:
                    cluster.append(candidate)
                    last_box = candidate

        if _cluster_is_valid(cluster, icon_size, settings):
            cluster_score = sum(float(b[4]) for b in cluster)
            if len(cluster) > len(best_cluster) or (
                    len(cluster) == len(best_cluster) and cluster_score > best_score
            ):
                best_cluster = cluster
                best_score = cluster_score

    if best_cluster:
        return best_cluster

    return [max(portrait_boxes, key=lambda b: float(b[4]))]

# -------------------- Discovery Phase --------------------


def run_discovery_phase(
        video_path: str | Path,
        roi: Optional[tuple[float, float, float, float]],
        agent_templates: List[AgentTemplate],
        settings: Settings,
) -> DiscoveryResult:
    """
    Discovery Phase: находим активных агентов, оптимальный scale и Anchor Zone.
    """

    if not agent_templates:
        raise ValueError("M4 – Discovery phase called with empty agent_templates")

    logger.info(
        "M4 – starting Discovery Phase (%.1f FPS full-video scan, preprocess=%s)...",
        settings.cv_discovery_fps,
        settings.cv_preprocess_mode,
    )

    primary_scales = np.linspace(0.03, 0.15, num=15)
    fallback_scales = np.linspace(0.03, 0.30, num=15)

    detections_by_agent: Dict[str, List[Tuple[float, int, int, int, int]]] = {}
    template_size = agent_templates[0].image.shape[0]
    last_frame_h: Optional[int] = None

    for ef in iter_frames(
            video_path, roi, target_fps=settings.cv_discovery_fps, debug_frames=False
    ):
        frame = ef.frame
        if frame.size == 0:
            continue

        last_frame_h = frame.shape[0]
        found_in_frame = False
        processed_frame = preprocess_for_template_matching(frame, settings)

        for scales in (primary_scales, fallback_scales):
            for scale in scales:
                new_size = max(1, int(template_size * scale))
                resized_templates = []
                for at in agent_templates:
                    resized = cv2.resize(
                        at.image, (new_size, new_size), interpolation=cv2.INTER_AREA
                    )
                    resized_templates.append(
                        preprocess_for_template_matching(resized, settings)
                    )

                matches = match_templates(
                    processed_frame,
                    resized_templates,
                    threshold=settings.cv_discovery_threshold,
                )
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

    active_agents_info = {
        name: detections
        for name, detections in detections_by_agent.items()
        if len(detections) >= settings.cv_discovery_min_detections
    }

    if not active_agents_info:
        logger.warning(
            "M4 – Discovery failed: no agents with >= %d detections, falling back",
            settings.cv_discovery_min_detections,
        )
        frame_h = last_frame_h or 100
        fallback_size_px = max(1, int(frame_h * settings.cv_fallback_size_percent))
        optimal_scale = fallback_size_px / template_size
        return DiscoveryResult(
            active_agents={at.name for at in agent_templates},
            median_scale=optimal_scale,
            anchor_zone=None,
            stats={},
        )

    active_agent_names = sorted(list(active_agents_info.keys()))
    all_scales = [d[0] for detections in active_agents_info.values() for d in detections]
    optimal_scale = float(np.median(all_scales))

    all_x = [d[1] for detections in active_agents_info.values() for d in detections]
    all_y = [d[2] for detections in active_agents_info.values() for d in detections]
    all_w = [d[3] for detections in active_agents_info.values() for d in detections]
    all_h = [d[4] for detections in active_agents_info.values() for d in detections]

    min_x = min(all_x)
    min_y = min(all_y)
    max_x = max([x + w for x, w in zip(all_x, all_w)])
    max_y = max([y + h for y, h in zip(all_y, all_h)])

    icon_size = max(1, int(template_size * optimal_scale))
    margin_x = int(icon_size * settings.cv_anchor_margin_x)
    margin_y = int(icon_size * settings.cv_anchor_margin_y)

    anchor_zone = (
        max(0, int(min_x - margin_x)),
        max(0, int(min_y - margin_y)),
        int(max_x + margin_x),
        int(max_y + margin_y),
    )

    stats = {}
    for name, detections in active_agents_info.items():
        stats[name] = {
            "count": len(detections),
            "scales": [d[0] for d in detections],
            "bboxes": [(d[1], d[2], d[3], d[4]) for d in detections],
        }

    coverage = len(active_agent_names) / len(agent_templates)
    logger.info(
        "M4 – Discovery Phase COMPLETE: active=%s, coverage=%.1f%%, "
        "scale=%.3f, icon_size=%dpx, Anchor Zone=%s",
        active_agent_names,
        coverage * 100,
        optimal_scale,
        icon_size,
        anchor_zone,
        )

    return DiscoveryResult(
        active_agents=set(active_agent_names),
        median_scale=optimal_scale,
        anchor_zone=anchor_zone,
        stats=stats,
    )


# -------------------- Основной проход --------------------


def detect_speakers(
        video_path: str | Path,
        roi: Optional[tuple[float, float, float, float]],
        target_fps: int,
        agent_templates_dir: str | Path,
        settings: Settings,
        debug_frames: bool = False,
        debug_frames_dir: Path | str = "debug_frames",
) -> tuple[List[CVDetection], int]:
    """
    Полный проход CV-портрет детектора по видео.
    """

    agents_list = None
    if settings.cv_agents_list:
        agents_list = [a.strip() for a in settings.cv_agents_list.split(",")]

    agent_templates: List[AgentTemplate] = load_agent_templates(
        agent_templates_dir,
        agents_list=agents_list,
    )

    if not agent_templates:
        logger.error("No agent templates found in %s", agent_templates_dir)
        raise ValueError(f"No agent templates found in {agent_templates_dir}")

    logger.info("M4 – preprocess mode: %s", settings.cv_preprocess_mode)
    logger.info(
        "M4 – geometric clustering: h_tol=%.1f, v_min=%.1f, v_max=%.1f (× icon_size)",
        settings.cv_cluster_horizontal_tol,
        settings.cv_cluster_vertical_min,
        settings.cv_cluster_vertical_max,
    )

    if settings.cv_skip_discovery:
        logger.info("M4 – skipping Discovery Phase as per settings")
        discovery = DiscoveryResult(
            active_agents={at.name for at in agent_templates},
            median_scale=-1.0,
            anchor_zone=None,
            stats={},
        )
    else:
        discovery = run_discovery_phase(
            video_path=video_path,
            roi=roi,
            agent_templates=agent_templates,
            settings=settings,
        )

    active_templates = [t for t in agent_templates if t.name in discovery.active_agents]
    optimal_scale = discovery.median_scale
    anchor_zone = discovery.anchor_zone

    agent_images: Optional[List[np.ndarray]] = None
    template_size = active_templates[0].image.shape[0]
    icon_size = max(1, int(template_size * optimal_scale)) if optimal_scale > 0 else 0

    if optimal_scale > 0:
        new_size = max(1, int(template_size * optimal_scale))
        icon_size = new_size
        agent_images = []
        for at in active_templates:
            resized = cv2.resize(
                at.image, (new_size, new_size), interpolation=cv2.INTER_AREA
            )
            agent_images.append(preprocess_for_template_matching(resized, settings))

    raw_frame_detections: List[Dict[str, float]] = []
    timestamps: List[float] = []
    total_frames: int = 0

    if debug_frames:
        debug_dir = Path(debug_frames_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv_debug_log_path = debug_dir / "cv_detections.jsonl"
        cv_debug_log = open(cv_debug_log_path, "w", encoding="utf-8")
    else:
        cv_debug_log = None

    for ef in iter_frames(video_path, roi, target_fps, debug_frames=False):
        ts = ef.timestamp
        frame = ef.frame
        if frame.size == 0:
            continue

        total_frames += 1

        if agent_images is None:
            if optimal_scale <= 0:
                frame_h = frame.shape[0]
                fallback_size_px = max(
                    1, int(frame_h * settings.cv_fallback_size_percent)
                )
                optimal_scale = fallback_size_px / template_size

            new_size = max(1, int(template_size * optimal_scale))
            icon_size = new_size
            agent_images = []
            for at in active_templates:
                resized = cv2.resize(
                    at.image, (new_size, new_size), interpolation=cv2.INTER_AREA
                )
                agent_images.append(preprocess_for_template_matching(resized, settings))

        search_frame = frame
        offset_x, offset_y = 0, 0
        if anchor_zone:
            x1, y1, x2, y2 = anchor_zone
            x1 = max(0, min(x1, frame.shape[1] - 1))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            x2 = max(x1 + 1, min(x2, frame.shape[1]))
            y2 = max(y1 + 1, min(y2, frame.shape[0]))
            search_frame = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1

        processed_search_frame = preprocess_for_template_matching(search_frame, settings)
        portrait_boxes = match_templates(
            processed_search_frame,
            agent_images,
            threshold=settings.cv_portrait_threshold,
        )
        portrait_boxes = non_max_suppression(portrait_boxes)

        if portrait_boxes and icon_size > 0:
            before_count = len(portrait_boxes)
            portrait_boxes = _find_best_vertical_cluster(
                portrait_boxes,
                icon_size,
                settings,
            )
            after_count = len(portrait_boxes)
            if before_count != after_count:
                logger.debug(
                    "M4 – geometric filter frame=%d reduced detections %d -> %d",
                    ef.index,
                    before_count,
                    after_count,
                )

        if anchor_zone:
            translated_boxes = []
            for x, y, w_b, h_b, s, p_idx in portrait_boxes:
                translated_boxes.append(
                    (x + offset_x, y + offset_y, w_b, h_b, s, p_idx)
                )
            portrait_boxes = translated_boxes

        active_this_frame: Dict[str, float] = {}
        for pb in portrait_boxes:
            px, py, pw, ph, ps, p_idx = pb
            agent_name = active_templates[p_idx].name
            active_this_frame[agent_name] = max(
                active_this_frame.get(agent_name, 0.0), ps
            )

        raw_frame_detections.append(active_this_frame)
        timestamps.append(ts)

        if cv_debug_log:
            log_entry = {
                "timestamp_ms": int(ts * 1000),
                "frame_index": ef.index,
                "detections": active_this_frame,
            }
            cv_debug_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        if debug_frames:
            debug_img = frame.copy()
            if anchor_zone:
                x1, y1, x2, y2 = anchor_zone
                x1 = max(0, min(x1, frame.shape[1] - 1))
                y1 = max(0, min(y1, frame.shape[0] - 1))
                x2 = max(x1 + 1, min(x2, frame.shape[1]))
                y2 = max(y1 + 1, min(y2, frame.shape[0]))
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

            for pb in portrait_boxes:
                x, y, w_b, h_b, s, p_idx = pb
                cv2.rectangle(debug_img, (x, y), (x + w_b, y + h_b), (255, 0, 0), 2)
                label = f"{active_templates[p_idx].name} ({s:.2f})"
                cv2.putText(
                    debug_img,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

            ts_ms = int(ts * 1000)
            dest = Path(debug_frames_dir) / f"frame_{ef.index:06d}_{ts_ms:08d}.png"
            cv2.imwrite(str(dest), debug_img)

    if cv_debug_log:
        cv_debug_log.close()

    # Temporal smoothing K-of-N
    k = settings.cv_temporal_k
    n = settings.cv_temporal_n
    smoothed_detections: List[CVDetection] = []
    all_agent_names = {at.name for at in active_templates}

    for agent_name in all_agent_names:
        history = [0] * len(raw_frame_detections)
        for i, detections in enumerate(raw_frame_detections):
            if agent_name in detections:
                history[i] = 1

        for i in range(len(history)):
            window = history[max(0, i - n + 1) : i + 1]
            if sum(window) >= k:
                conf = raw_frame_detections[i].get(agent_name, 0.0)
                if conf == 0.0:
                    detected_confs = [
                        raw_frame_detections[j][agent_name]
                        for j in range(max(0, i - n + 1), i + 1)
                        if agent_name in raw_frame_detections[j]
                    ]
                    conf = (
                        sum(detected_confs) / len(detected_confs)
                        if detected_confs
                        else 0.5
                    )

                smoothed_detections.append(
                    CVDetection(
                        timestamp=timestamps[i],
                        agent_name=agent_name,
                        confidence=conf,
                    )
                )

    smoothed_detections.sort(key=lambda x: x.timestamp)
    logger.info(
        "M4 – detected %d speaking events across %d frames (after smoothing)",
        len(smoothed_detections),
        total_frames,
    )
    return smoothed_detections, total_frames
