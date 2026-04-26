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
    agent_img = np.zeros((100, 100, 3), dtype=np.uint8)
    agent_img[20:80, 20:80] = 255
    cv2.imwrite(str(templates_dir / "jett.png"), agent_img)
    
    # Mock iter_frames to return a frame with the agent
    class MockFrame:
        def __init__(self, index=0, timestamp=0.1):
            self.index = index
            self.timestamp = timestamp
            # Larger frame containing the template
            self.frame = np.zeros((300, 300, 3), dtype=np.uint8)
            # Put it at 50x50 size (which is 50/512 = ~0.097 scale of normalized 512x512)
            # Actually, since load_agent_templates normalizes to 512x512, 
            # we should expect it to be matched at some scale.
            scaled = cv2.resize(agent_img, (50, 50))
            self.frame[10:60, 10:60] = scaled
            
    def mock_iter_frames(*args, **kwargs):
        # Need multiple frames for discovery to succeed if min_detections > 1
        yield MockFrame(index=0, timestamp=0.1)
        yield MockFrame(index=1, timestamp=0.2)
        yield MockFrame(index=2, timestamp=0.3)
        yield MockFrame(index=3, timestamp=0.4)
        yield MockFrame(index=4, timestamp=0.5)
        
    monkeypatch.setattr("app.services.m4_cv_mic_detector.iter_frames", mock_iter_frames)
    
    settings = Settings()
    settings.cv_temporal_k = 1
    settings.cv_temporal_n = 1
    settings.cv_portrait_threshold = 0.5
    settings.cv_discovery_threshold = 0.5
    settings.cv_discovery_min_detections = 1 # Make it easy
    
    detections, total_frames = detect_speakers(
        video_path="dummy.mp4",
        roi=None,
        target_fps=5,
        agent_templates_dir=templates_dir,
        settings=settings
    )
    
    # Discovery will run, then the main loop will run.
    # mock_iter_frames will be called twice (once for discovery, once for detection).
    
    assert total_frames == 5
    assert len(detections) >= 1
    assert any(d.agent_name == "jett" for d in detections)

def test_detect_speakers_calls_discovery_once(tmp_path, monkeypatch):
    templates_dir = tmp_path / "agents"
    templates_dir.mkdir()
    agent_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(templates_dir / "jett.png"), agent_img)

    discovery_calls = []
    
    from app.utils.cv_logic import AgentTemplate

    def mock_discovery(*args, **kwargs):
        discovery_calls.append(True)
        # Return something to avoid failure
        img = np.zeros((512, 512, 4), dtype=np.uint8)
        return [AgentTemplate(name="jett", image=img)], 0.1, (10, 10, 100, 100)

    monkeypatch.setattr("app.services.m4_cv_mic_detector.run_discovery_phase", mock_discovery)
    
    # Also need to mock iter_frames for the main loop
    class MockFrame:
        def __init__(self):
            self.index = 0
            self.timestamp = 0.1
            self.frame = np.zeros((300, 300, 3), dtype=np.uint8)

    def mock_iter_frames(*args, **kwargs):
        yield MockFrame()

    monkeypatch.setattr("app.services.m4_cv_mic_detector.iter_frames", mock_iter_frames)

    settings = Settings()
    detect_speakers(
        video_path="dummy.mp4",
        roi=None,
        target_fps=5,
        agent_templates_dir=templates_dir,
        settings=settings
    )
    assert len(discovery_calls) == 1
