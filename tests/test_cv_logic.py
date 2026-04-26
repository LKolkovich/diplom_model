"""Tests for CV helper utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.utils.cv_logic import (
    compute_phash,
    identify_agent,
    load_agent_templates,
    match_template,
    phash_distance,
)


def test_phash_same_image_distance_zero() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    h1 = compute_phash(img)
    h2 = compute_phash(img)
    assert phash_distance(h1, h2) == 0


def test_phash_different_images_nonzero() -> None:
    rng = np.random.default_rng(1)
    img1 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    img2 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    h1 = compute_phash(img1)
    h2 = compute_phash(img2)
    assert phash_distance(h1, h2) > 0


def test_match_template_finds_subimage() -> None:
    scene = np.zeros((200, 200, 3), dtype=np.uint8)
    scene[50:70, 80:100] = 255
    template = scene[50:70, 80:100].copy()
    match = match_template(scene, template, threshold=0.9)
    assert match is not None
    assert match.score > 0.9


def test_match_template_no_match() -> None:
    rng = np.random.default_rng(99)
    scene = rng.integers(50, 100, (100, 100, 3), dtype=np.uint8)
    template = rng.integers(200, 255, (20, 20, 3), dtype=np.uint8)
    match = match_template(scene, template, threshold=0.95)
    assert match is None


def test_load_agent_templates_empty_dir(tmp_path: Path) -> None:
    templates = load_agent_templates(tmp_path)
    assert templates == []


def test_load_agent_templates(agent_template_dir: Path) -> None:
    templates = load_agent_templates(agent_template_dir)
    assert len(templates) == 3
    names = {t.name for t in templates}
    assert names == {"jett", "reyna", "sage"}
    for t in templates:
        assert t.image.shape == (512, 512, 4)


def test_identify_agent_returns_closest(agent_template_dir: Path) -> None:
    templates = load_agent_templates(agent_template_dir)
    candidate = templates[0].image.copy()
    agent, confidence = identify_agent(candidate, templates, phash_threshold=20)
    assert agent == templates[0].name
    assert confidence > 0.5


def test_identify_agent_no_templates() -> None:
    img = np.zeros((36, 36, 3), dtype=np.uint8)
    agent, confidence = identify_agent(img, [], phash_threshold=10)
    assert agent is None
    assert confidence == 0.0
