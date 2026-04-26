# Valorant Subtitle Generation Model

A FastAPI-based pipeline that transcribes and diarises Valorant in-game voice
chat recordings, then exports speaker-labelled SRT subtitle files.

## Architecture

```
POST /tasks
     │
     ▼
M0  Orchestrator
     │
     ├─► M1  AudioValidator     – converts source file to 16 kHz mono WAV
     │
     ├─► M2  FrameExtractor     – samples frames (ROI or full) from the video
     │
     ├─► M3  WhisperXASR        – transcription + alignment + diarization
     │
     ├─► M4  CVMicDetector      – template matching + NMS for active speaker portraits
     │
     ├─► M5  FusionEngine       – maps generic speaker IDs → agent names (Fusion Logic)
     │
     └─► M6  SRTExporter        – writes per-agent + combined SRT files
```

## Quick Start

```bash
# 1. Clone & install
pip install -r requirements.txt

# 2. Copy environment config
cp .env.example .env
# Edit .env – set HF_TOKEN (required for diarization) and DEVICE

# 3. Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Submit a job (multipart/form-data)
curl -X POST http://localhost:8000/tasks \
     -F "video=@/path/to/game.mp4" \
     -F 'config={"video_mode": "full_frame", "language": "en"}'

# 5. Poll for progress
curl http://localhost:8000/tasks/<task_id>

# 6. Download the result ZIP
curl -O http://localhost:8000/tasks/<task_id>/result
```

## CV Detection System

The CV system (M4) detects speaking agents by matching templates for agent portraits. M4 automatically calibrates portrait template size for the current video resolution and ROI before the main detection pass. This allows one template set to work across different resolutions without manual resizing.

### Templates
Place templates in the following directory:
- `templates/agents/`: Agent portrait images (e.g., `jett.png`, `sage.png`). Templates can be prepared in any arbitrary convenient size.

### Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `VIDEO_MODE` | `full_frame` | `full_frame` (use `VOICECHAT_ROI`) or `user_crop` |
| `VOICECHAT_ROI` | `0.0,0.15,0.075,0.65` | Fractional ROI (x1,y1,x2,y2) for `full_frame` |
| `CV_PORTRAIT_THRESHOLD` | `0.7` | Matching threshold for agent portraits |
| `CV_TEMPORAL_K` | `3` | Minimum detections in window for positive signal |
| `CV_TEMPORAL_N` | `5` | Sliding window size for temporal smoothing |
| `CV_CALIBRATION_MAX_FRAMES` | `30` | Max frames to examine during automatic calibration |
| `CV_FALLBACK_SIZE_PERCENT` | `0.08` | Fallback template height as % of ROI height |
| `FUSION_COVERAGE_THRESHOLD` | `0.5` | Min temporal overlap to map speaker to agent |
| `FUSION_CONFIDENCE_THRESHOLD` | `0.8` | Min CV confidence to map speaker to agent |

### Debug Artifacts

When `DEBUG_FRAMES=true` is set in `.env`:
- **Overlay Frames**: Saved to `debug_frames/frame_{idx}_{ts}.png`. Shows detected bounding boxes and agent names.
- **CV Detections Log**: `debug_frames/cv_detections.jsonl`. Per-frame raw detections with timestamps and confidence scores.

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/tasks` | Submit video and config, returns `task_id` |
| `GET` | `/tasks/{task_id}` | Get status, progress, and current module |
| `GET` | `/tasks/{task_id}/result` | Download result ZIP (completed tasks only) |
| `GET` | `/health` | Liveness probe |

Interactive API docs: `http://localhost:8000/docs`.

## Running Tests

To run the test suite, ensure you have the dev dependencies installed:

```bash
pip install pytest soundfile
pytest
```
