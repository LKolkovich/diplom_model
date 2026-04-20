"""CV helper utilities.

Provides:
- pHash computation and comparison (via imagehash).
- Template matching wrapper around OpenCV's matchTemplate.
- Agent template library loader.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


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
    candidate: np.ndarray,
    templates: List[AgentTemplate],
    phash_threshold: int = 10,
) -> Tuple[Optional[str], float]:
    """Identify an agent in *candidate* ROI by pHash comparison.

    Parameters
    ----------
    candidate:
        Cropped BGR image of the suspected agent portrait area.
    templates:
        Pre-loaded agent templates.
    phash_threshold:
        Maximum Hamming distance to count as a match.

    Returns
    -------
    (agent_name, confidence) or (None, 0.0)
        *confidence* is ``1 - hamming / 64`` normalised to [0, 1].
    """
    if not templates or candidate is None or candidate.size == 0:
        return None, 0.0

    candidate_hash = compute_phash(candidate)
    best_name: Optional[str] = None
    best_dist = phash_threshold + 1

    for tmpl in templates:
        dist = phash_distance(candidate_hash, tmpl.phash)
        if dist < best_dist:
            best_dist = dist
            best_name = tmpl.name

    if best_name is None:
        return None, 0.0

    confidence = max(0.0, 1.0 - best_dist / 64.0)
    return best_name, confidence
