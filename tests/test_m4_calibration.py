import pytest
import numpy as np
import cv2
from unittest.mock import patch
from app.services.m4_cv_mic_detector import calibrate_templates, _get_fallback_scale
from app.utils.cv_logic import AgentTemplate
from app.config import Settings
from app.services.m2_frame_extractor import ExtractedFrame

@pytest.fixture
def settings():
    return Settings(
        cv_portrait_threshold=0.7, 
        cv_calibration_max_frames=30, 
        cv_fallback_size_percent=0.08,
        templates_dir="tests/mock_templates" # dummy
    )

@pytest.fixture
def agent_template():
    # Create a 50x50 square as template
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (40, 40), (255, 255, 255), -1)
    return AgentTemplate(name="test_agent", image=img)

def test_calibration_success(settings, agent_template):
    # Scale = 1.2
    target_scale = 1.2
    ref_h, ref_w = agent_template.image.shape[:2]
    target_h, target_w = int(ref_h * target_scale), int(ref_w * target_scale)
    
    # Create a frame with the scaled template
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    scaled_template = cv2.resize(agent_template.image, (target_w, target_h))
    frame[50:50+target_h, 50:50+target_w] = scaled_template
    
    mock_frames = [ExtractedFrame(index=i, timestamp=i/5.0, frame=frame) for i in range(10)]
    
    with patch("app.services.m4_cv_mic_detector.iter_frames", return_value=iter(mock_frames)):
        scale = calibrate_templates("fake_path", None, 5, [agent_template], settings)
        
    # The coarse search uses 0.1 steps. 1.2 should be found.
    # The fine search should refine it.
    assert abs(scale - target_scale) < 0.05

def test_calibration_fallback(settings, agent_template):
    # Empty frame (black)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    
    mock_frames = [ExtractedFrame(index=i, timestamp=i/5.0, frame=frame) for i in range(10)]
    
    with patch("app.services.m4_cv_mic_detector.iter_frames", return_value=iter(mock_frames)):
        scale = calibrate_templates("fake_path", None, 5, [agent_template], settings)
    
    # ROI height is 200. Fallback is 8% = 16.
    # Template height is 50.
    # Fallback scale = 16 / 50 = 0.32
    expected_fallback = (200 * 0.08) / 50
    assert abs(scale - expected_fallback) < 0.01

def test_get_fallback_scale(settings):
    scale = _get_fallback_scale(1000, 100, settings)
    # 1000 * 0.08 / 100 = 80 / 100 = 0.8
    assert scale == 0.8
