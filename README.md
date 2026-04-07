# Valorant Subtitle Generation Model

A FastAPI-based pipeline that transcribes and diarises Valorant in-game voice
chat recordings, then exports speaker-labelled SRT subtitle files.

## Architecture

```
POST /process
     │
     ▼
M0  Orchestrator
     │
     ├─► M1  AudioValidator     – converts source file to 16 kHz mono WAV
     │
     ├─► M2  FrameExtractor     – samples ROI frames from the video
     │
     ├─► M3  WhisperXASR        – transcription + alignment + diarization
     │
     ├─► M4  CVMicDetector      – pHash + template matching for agent IDs
     │
     ├─► M5  FusionEngine       – maps generic speaker IDs → agent names
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

# 4. Submit a job
curl -s -X POST http://localhost:8000/process \
     -H "Content-Type: application/json" \
     -d '{"video_path": "/path/to/your/game.mp4"}' | python3 -m json.tool

# 5. Poll for progress
curl http://localhost:8000/status/<task_id>

# 6. Download the SRT files
curl http://localhost:8000/result/<task_id>
```

## Agent Portrait Templates

Place portrait PNG/JPEG images named after each agent (lowercase) in
`templates/agents/`:

```
templates/
  agents/
    jett.png
    reyna.png
    sage.png
    ...
  mic_active.png     ← speaking indicator icon (optional but improves accuracy)
```

When `mic_active.png` is absent, M4 falls back to a brightness-based
heuristic to detect activity.

## Configuration

All settings are controlled via environment variables (or `.env` file):

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | `` | HuggingFace token for pyannote diarization |
| `WHISPER_MODEL` | `large-v2` | Whisper model size |
| `DEVICE` | `cpu` | `cuda` or `cpu` |
| `COMPUTE_TYPE` | `int8` | Quantisation: `float16`, `int8_float16`, `int8` |
| `FRAME_RATE` | `5` | Frames per second to sample from video |
| `OUTPUT_DIR` | `output` | Directory for generated SRT files |
| `VOICECHAT_ROI` | `0.0,0.15,0.075,0.65` | Fractional ROI of voice panel |
| `PHASH_THRESHOLD` | `10` | Max pHash Hamming distance for agent match |

## Running Tests

```bash
pip install pytest soundfile
pytest
```

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/process` | Start pipeline, returns `task_id` |
| `GET` | `/status/{task_id}` | Progress 0-100 %, current module |
| `GET` | `/result/{task_id}` | SRT download links (when completed) |
| `GET` | `/download/{task_id}/{filename}` | Download a specific SRT |
| `GET` | `/health` | Liveness probe |

Interactive API docs are available at `http://localhost:8000/docs`.
