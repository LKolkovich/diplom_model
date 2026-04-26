"""CV helper utilities.

Provides:
- pHash computation and comparison (via imagehash).
- Template matching wrapper around OpenCV's matchTemplate.
- Agent template library loader.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import imagehash
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Template Matching & NMS
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    x: int
    y: int
    w: int
    h: int
    score: float


def match_templates(
    image: np.ndarray,
    templates: List[np.ndarray],
    threshold: float = 0.7,
) -> List[Tuple[int, int, int, int, float, int]]:
    """
    Find all occurrences of templates in image.
    Returns list of (x, y, w, h, score, template_index).
    """
    all_detections = []
    if image is None or image.size == 0:
        return []

    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    for t_idx, template in enumerate(templates):
        if template is None or template.size == 0:
            continue
        
        if len(template.shape) == 3:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray_template = template
            
        th, tw = gray_template.shape[:2]
        if th > gray_image.shape[0] or tw > gray_image.shape[1]:
            continue

        res = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        
        for pt in zip(*loc[::-1]): # x, y
            score = res[pt[1], pt[0]]
            all_detections.append((int(pt[0]), int(pt[1]), int(tw), int(th), float(score), t_idx))
            
    return all_detections


def match_template(
    image: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.7,
) -> Optional[MatchResult]:
    """Thin wrapper around match_templates for legacy support."""
    results = match_templates(image, [template], threshold)
    if not results:
        return None
    # Return best match
    best = max(results, key=lambda x: x[4])
    return MatchResult(x=best[0], y=best[1], w=best[2], h=best[3], score=best[4])


def non_max_suppression(
    boxes: List[Tuple[int, int, int, int, float, int]], 
    iou_threshold: float = 0.3
) -> List[Tuple[int, int, int, int, float, int]]:
    """
    Simple NMS to filter overlapping boxes.
    """
    if not boxes:
        return []

    # Sort by score descending
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        
        remaining = []
        for box in boxes:
            if compute_iou(best, box) < iou_threshold:
                remaining.append(box)
        boxes = remaining
        
    return keep

def compute_iou(boxA, boxB):
    # box = (x, y, w, h, score)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# ---------------------------------------------------------------------------
# pHash helpers
# ---------------------------------------------------------------------------

def compute_phash(image: np.ndarray, hash_size: int = 8) -> imagehash.ImageHash:
    """Compute a perceptual hash for a BGR or greyscale *image* array."""
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil, hash_size=hash_size)


def phash_distance(a: imagehash.ImageHash, b: imagehash.ImageHash) -> int:
    """Hamming distance between two pHashes."""
    return int(a - b)


def load_simple_templates(templates_dir: str | Path) -> List[np.ndarray]:
    """Load all images from *templates_dir* as a list of numpy arrays."""
    dirpath = Path(templates_dir)
    templates: List[np.ndarray] = []

    if not dirpath.exists():
        logger.warning("Templates directory '%s' does not exist.", dirpath)
        return templates

    for path in sorted(dirpath.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            img = cv2.imread(str(path))
            if img is None:
                logger.warning("Could not read template image: %s", path)
                continue
            templates.append(img)
            logger.debug("Loaded template from %s", path.name)

    logger.info("Loaded %d templates from '%s'", len(templates), dirpath)
    return templates


# ---------------------------------------------------------------------------
# Agent template library
# ---------------------------------------------------------------------------

class AgentTemplate:
    """Holds the BGR image and precomputed pHash for one agent portrait."""

    __slots__ = ("name", "image", "phash")

    def __init__(self, name: str, image: np.ndarray) -> None:
        self.name = name
        self.image = image
        self.phash = compute_phash(image)


def load_agent_templates(templates_dir: str | Path) -> List[AgentTemplate]:
    """Load all agent portrait images from *templates_dir*.

    Each image file should be named after the agent, e.g. ``jett.png``,
    ``reyna.png``.  Subdirectories are ignored.

    Returns
    -------
    list[AgentTemplate]
        Sorted by agent name for deterministic ordering.
    """
    dirpath = Path(templates_dir)
    templates: List[AgentTemplate] = []

    if not dirpath.exists():
        logger.warning("Templates directory '%s' does not exist.", dirpath)
        return templates

    for path in sorted(dirpath.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            img = cv2.imread(str(path))
            if img is None:
                logger.warning("Could not read template image: %s", path)
                continue
            agent_name = path.stem.lower()
            templates.append(AgentTemplate(name=agent_name, image=img))
            logger.debug("Loaded template '%s' from %s", agent_name, path.name)

    logger.info("Loaded %d agent templates from '%s'", len(templates), dirpath)
    return templates


def identify_agent(
    image: np.ndarray,
    templates: List[AgentTemplate],
    phash_threshold: int = 10,
) -> tuple[Optional[str], float]:
    """Identify which agent is in the image using pHash.
    
    Legacy abstraction used by some tests.
    """
    if not templates:
        return None, 0.0
    
    target_hash = compute_phash(image)
    best_name = None
    min_dist = float('inf')
    
    for t in templates:
        dist = phash_distance(target_hash, t.phash)
        if dist < min_dist:
            min_dist = dist
            best_name = t.name
            
    if min_dist <= phash_threshold:
        # Map distance to a 0-1 confidence score
        # dist=0 -> 1.0, dist=phash_threshold -> 0.5?
        confidence = 1.0 - (min_dist / (phash_threshold * 2))
        return best_name, confidence
    
    return None, 0.0
