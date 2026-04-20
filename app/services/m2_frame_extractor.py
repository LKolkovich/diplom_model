"""M2 – Frame Extractor.

Extracts frames from a video file at a configurable frame-rate and crops each
frame to the Region-of-Interest (ROI) covering the Valorant voice-chat panel
(left edge of the screen by default).

The ROI is expressed as fractional coordinates (x1, y1, x2, y2) in [0, 1]
so the same configuration works for any source resolution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFrame:
    index: int
    timestamp: float
    frame: np.ndarray


def _open_capture(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    return cap


def _crop_roi(frame: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = int(roi[0] * w)
    y1 = int(roi[1] * h)
    x2 = int(roi[2] * w)
    y2 = int(roi[3] * h)
    x1, x2 = sorted([max(0, x1), min(w, x2)])
    y1, y2 = sorted([max(0, y1), min(h, y2)])
    return frame[y1:y2, x1:x2]


def iter_frames(
    video_path: str | Path,
    roi: Tuple[float, float, float, float],
    target_fps: int = 5,
    debug_frames: bool = False,
    debug_frames_dir: Path | str = "debug_frames",
) -> Iterator[ExtractedFrame]:
    """Yield ROI-cropped frames at *target_fps* frames per second.

    Parameters
    ----------
    video_path:
        Path to the source video.
    roi:
        Fractional (x1, y1, x2, y2) region of interest.
    target_fps:
        How many frames per second to sample.
    debug_frames:
        If True, save extracted frames as PNG files to *debug_frames_dir*.
    debug_frames_dir:
        Directory to save debug PNG files.

    Yields
    ------
    ExtractedFrame
        Named tuple with *index*, *timestamp* (seconds) and *frame* array.
    """
    path = Path(video_path)
    cap = _open_capture(path)

    native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(native_fps / target_fps))

    logger.info(
        "M2 – extracting frames from '%s' (native %.1f fps → every %d frames, ~%d fps)",
        path.name,
        native_fps,
        frame_interval,
        target_fps,
    )

    if debug_frames:
        debug_dir = Path(debug_frames_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info("M2 – debug frames enabled, saving to %s", debug_dir)

    frame_idx = 0
    sampled_idx = 0

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / native_fps
                roi_frame = _crop_roi(bgr, roi)

                if debug_frames:
                    # Save as PNG
                    dest = Path(debug_frames_dir) / f"frame_{sampled_idx:06d}_{timestamp:.3f}s.png"
                    cv2.imwrite(str(dest), roi_frame)

                yield ExtractedFrame(index=sampled_idx, timestamp=timestamp, frame=roi_frame)
                sampled_idx += 1

            frame_idx += 1
    finally:
        cap.release()

    logger.info("M2 – extracted %d frames total", sampled_idx)


def extract_frames_to_disk(
    video_path: str | Path,
    output_dir: str | Path,
    roi: Tuple[float, float, float, float],
    target_fps: int = 5,
) -> list[Path]:
    """Write cropped frames as JPEG files and return their paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for ef in iter_frames(video_path, roi, target_fps):
        dest = out / f"frame_{ef.index:06d}_{ef.timestamp:.3f}s.jpg"
        cv2.imwrite(str(dest), ef.frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(dest)

    return paths
