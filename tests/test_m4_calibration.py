import pytest
import numpy as np
import cv2
from unittest.mock import patch
from app.services.m4_cv_mic_detector import calibrate_templates
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
    # Create a 100x100 square as template
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
    return AgentTemplate(name="test_agent", image=img)

def test_calibrate_templates_success(settings, agent_template):
    # Given:
    # - template size 100x100
    # - frame contains matching portrait at ~42x42
    target_size = 42
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    scaled_template = cv2.resize(agent_template.image, (target_size, target_size))
    frame[50:50+target_size, 50:50+target_size] = scaled_template
    
    mock_frames = [ExtractedFrame(index=0, timestamp=0.0, frame=frame)]
    
    with patch("app.services.m4_cv_mic_detector.iter_frames", return_value=iter(mock_frames)):
        resized_templates, size = calibrate_templates("fake_path", None, 5, [agent_template], settings)
        
    assert 40 <= size <= 44
    assert all(t.shape[:2] == (size, size) for t in resized_templates)

def test_calibrate_templates_fallback(settings, agent_template):
    # Blank frames only
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_frames = [ExtractedFrame(index=0, timestamp=0.0, frame=frame)]
    
    with patch("app.services.m4_cv_mic_detector.iter_frames", return_value=iter(mock_frames)):
        resized_templates, size = calibrate_templates("fake_path", None, 5, [agent_template], settings)
    
    # 200 * 0.08 = 16
    assert size == 16
    assert all(t.shape[:2] == (size, size) for t in resized_templates)

def test_calibrate_templates_empty_templates_raises(settings):
    with pytest.raises(ValueError, match="No agent templates provided for calibration"):
        calibrate_templates("fake_path", None, 5, [], settings)

def test_calibrate_templates_already_correct_size(settings, agent_template):
    # template is 100x100, frame has it at 100x100
    target_size = 100
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    frame[10:110, 10:110] = agent_template.image
    
    mock_frames = [ExtractedFrame(index=0, timestamp=0.0, frame=frame)]
    
    with patch("app.services.m4_cv_mic_detector.iter_frames", return_value=iter(mock_frames)):
        resized_templates, size = calibrate_templates("fake_path", None, 5, [agent_template], settings)
    
    assert 98 <= size <= 102
    assert all(t.shape[:2] == (size, size) for t in resized_templates)
