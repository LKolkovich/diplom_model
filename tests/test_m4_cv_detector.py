import pytest
from pathlib import Path
import numpy as np
import cv2
from app.services.m4_cv_mic_detector import detect_speakers
from app.config import Settings

def test_detect_speakers_portrait_only(tmp_path, monkeypatch):
    # Setup mock templates
    templates_dir = tmp_path / "agents"
    templates_dir.mkdir()
    
    # Create a dummy agent template (white square)
    agent_img = np.zeros((20, 20, 3), dtype=np.uint8)
    agent_img[5:15, 5:15] = 255
    cv2.imwrite(str(templates_dir / "jett.png"), agent_img)
    
    # Mock iter_frames to return a frame with the agent
    class MockFrame:
        def __init__(self):
            self.index = 0
            self.timestamp = 0.1
            # Larger frame containing the template
            self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
            self.frame[10:30, 10:30] = 255
            
    def mock_iter_frames(*args, **kwargs):
        yield MockFrame()
        
    monkeypatch.setattr("app.services.m4_cv_mic_detector.iter_frames", mock_iter_frames)
    
    settings = Settings()
    settings.cv_temporal_k = 1
    settings.cv_temporal_n = 1
    settings.cv_portrait_threshold = 0.5
    
    # M4 now only needs agent_templates_dir
    detections, total_frames = detect_speakers(
        video_path="dummy.mp4",
        roi=None,
        target_fps=5,
        agent_templates_dir=templates_dir,
        settings=settings
    )
    
    assert total_frames == 1
    assert len(detections) == 1
    assert detections[0].agent_name == "jett"
    assert detections[0].confidence > 0.5
