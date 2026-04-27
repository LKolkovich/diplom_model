import pytest
import numpy as np
import cv2
from unittest.mock import patch
from app.services.m4_cv_mic_detector import run_discovery_phase
from app.utils.cv_logic import AgentTemplate
from app.config import Settings
from app.services.m2_frame_extractor import ExtractedFrame

@pytest.fixture
def settings():
    return Settings(
        cv_portrait_threshold=0.7, 
        cv_discovery_threshold=0.65,
        cv_discovery_min_detections=1,
        cv_discovery_fps=1.0,
        templates_dir="tests/mock_templates" # dummy
    )

@pytest.fixture
def agent_template():
    # Create a 512x512 square as template (as per new normalization)
    img = np.zeros((512, 512, 4), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (412, 412), (255, 255, 255, 255), -1)
    return AgentTemplate(name="test_agent", image=img)

def test_run_discovery_phase_success(settings, agent_template):
    # Given:
    # - template size 512x512
    # - frame contains matching portrait at scale 0.1
    scale = 0.1
    target_size = int(512 * scale)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Resize template to target size for the frame
    # Convert template to 3 channels for the frame
    gray_template = cv2.cvtColor(agent_template.image, cv2.COLOR_BGRA2GRAY)
    scaled_template = cv2.resize(gray_template, (target_size, target_size))
    frame[50:50+target_size, 50:50+target_size] = cv2.cvtColor(scaled_template, cv2.COLOR_GRAY2BGR)
    
    mock_frames = [ExtractedFrame(index=0, timestamp=0.0, frame=frame)]
    
    with patch("app.services.m4_cv_mic_detector.iter_frames", return_value=iter(mock_frames)):
        discovery = run_discovery_phase("fake_path", None, [agent_template], settings)
        
    assert "test_agent" in discovery.active_agents
    assert 0.09 <= discovery.median_scale <= 0.11
    assert discovery.anchor_zone is not None
    # anchor zone should be around (50-5, 50-5, 50+51+50, 50+51+50) -> (45, 45, 151, 151)
    # wait, target_size is 51. 50+51 = 101. 101+50 = 151.
    assert discovery.anchor_zone[0] == 45
    assert discovery.anchor_zone[1] == 45
    assert discovery.anchor_zone[2] >= 150
    assert discovery.anchor_zone[3] >= 150

def test_run_discovery_phase_failed(settings, agent_template):
    # Blank frames only
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_frames = [ExtractedFrame(index=0, timestamp=0.0, frame=frame)]
    
    with patch("app.services.m4_cv_mic_detector.iter_frames", return_value=iter(mock_frames)):
        discovery = run_discovery_phase("fake_path", None, [agent_template], settings)
    
    assert "test_agent" in discovery.active_agents # Returns all templates if failed
    # fallback scale based on ROI height (200) and fallback % (0.08)
    # fallback_size = 200 * 0.08 = 16
    # optimal_scale = 16 / 512 = 0.03125
    assert 0.03 <= discovery.median_scale <= 0.04
    assert discovery.anchor_zone is None
