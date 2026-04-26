# Docker Usage Guide

This document describes how to build and run the Valorant Pipeline using Docker.

## Building the Image

By default, the build uses Docker BuildKit and a cache mount for `pip` to speed up subsequent builds.

### Standard Build (Cached)

```bash
DOCKER_BUILDKIT=1 docker build -t valorant-pipeline .
```

The first build will download all dependencies, but subsequent builds will reuse the `pip` download cache if `requirements.txt` hasn't changed, or only download new dependencies.

### Clean Rebuild (No Cache)

If you need to force a clean rebuild from scratch (e.g., to troubleshoot dependency issues):

```bash
DOCKER_BUILDKIT=1 docker build --no-cache -t valorant-pipeline .
```

## Running the CLI

You can run the CLI through Docker by mounting your local directories and providing the necessary environment variables.

### Full Pipeline

**Bash (Linux/macOS):**
```bash
docker run --rm \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/output:/workspace/output \
  -e HF_TOKEN="your_huggingface_token" \
  valorant-pipeline \
  python -m app.cli run_pipeline /workspace/data/video.mp4 --hf-token "your_huggingface_token"
```

**PowerShell (Windows):**
```powershell
docker run --rm `
  -v ${PWD}/data:/workspace/data `
  -v ${PWD}/output:/workspace/output `
  -e HF_TOKEN="your_huggingface_token" `
  valorant-pipeline `
  python -m app.cli run_pipeline /workspace/data/video.mp4 --hf-token "your_huggingface_token"
```

### Frame Extraction (M2)

**Bash:**
```bash
docker run --rm \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/output:/workspace/output \
  valorant-pipeline \
  python -m app.cli run_m2 /workspace/data/video.mp4 /workspace/output/frames --fps 5
```

**PowerShell:**
```powershell
docker run --rm `
  -v ${PWD}/data:/workspace/data `
  -v ${PWD}/output:/workspace/output `
  valorant-pipeline `
  python -m app.cli run_m2 /workspace/data/video.mp4 /workspace/output/frames --fps 5
```

### CV Portrait Detection (M4)

**Bash:**
```bash
docker run --rm \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/output:/workspace/output \
  valorant-pipeline \
  python -m app.cli run_m4 /workspace/data/video.mp4 /workspace/output/detections.json --debug-dir /workspace/output/debug
```

**PowerShell:**
```powershell
docker run --rm `
  -v ${PWD}/data:/workspace/data `
  -v ${PWD}/output:/workspace/output `
  valorant-pipeline `
  python -m app.cli run_m4 /workspace/data/video.mp4 /workspace/output/detections.json --debug-dir /workspace/output/debug
```

## Running the API

If you wish to run the FastAPI server, you must override the default CMD:

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/output:/workspace/output \
  -e HF_TOKEN="your_huggingface_token" \
  --name valorant-api \
  valorant-pipeline \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## Running Tests

```bash
docker run --rm valorant-pipeline pytest
```

## CLI Reference

### Subcommands

- `run_pipeline`: Runs the complete M1-M6 pipeline.
- `run_m2`: Runs only frame extraction and saves frames to disk.
- `run_m4`: Runs CV portrait detection, outputs a JSON of detections, and optionally saves debug overlay frames.

### Video Modes

- `full_frame` (default): Requires ROI cropping. The pipeline uses the `voicechat_roi` setting to extract the voice chat area from a full screen recording.
- `user_crop`: The video is already cropped to the voice chat area. No ROI crop is applied.

### ROI Argument

Both `run_pipeline`, `run_m2`, and `run_m4` support an optional `--roi` argument (e.g., `--roi 0.0,0.15,0.075,0.65`) which overrides the default settings when `video_mode` is `full_frame`.
