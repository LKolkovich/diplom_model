# Docker Usage Guide

This document describes how to build and run the Valorant Pipeline using Docker.

## Building the Image

```bash
docker build -t valorant-pipeline .
```

## Running the CLI

You can run the CLI through Docker by mounting your local data directory and providing the necessary environment variables.

### Full Pipeline

```bash
docker run --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/output:/workspace/output \
  -e HF_TOKEN="your_huggingface_token" \
  valorant-pipeline \
  python app/cli.py full /data/video.mp4 --hf-token "your_huggingface_token"
```

### Frame Extraction (M2)

```bash
docker run --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/output:/workspace/output \
  valorant-pipeline \
  python app/cli.py run_m2 /data/video.mp4 /workspace/output/frames --fps 5
```

### CV Mic Detection (M4)

```bash
docker run --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/output:/workspace/output \
  valorant-pipeline \
  python app/cli.py run_m4 /data/video.mp4 /workspace/output/detections.json --debug-dir /workspace/output/debug
```

## Running the API

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/output:/workspace/output \
  -e HF_TOKEN="your_huggingface_token" \
  --name valorant-api \
  valorant-pipeline
```

The API will be available at `http://localhost:8000`.

## Running Tests

```bash
docker run --rm valorant-pipeline pytest
```

## CLI Reference

### Subcommands

- `full`: Runs the complete M1-M6 pipeline.
- `run_m2`: Runs only frame extraction and saves frames to disk.
- `run_m4`: Runs CV mic detection, outputs a JSON of detections, and optionally saves debug overlay frames.

### Video Modes

- `full_frame` (default): Requires ROI cropping. The pipeline uses the `voicechat_roi` setting to extract the voice chat area from a full screen recording.
- `user_crop`: The video is already cropped to the voice chat area. No ROI crop is applied.

### ROI Argument

Both `full`, `run_m2`, and `run_m4` support an optional `--roi` argument (e.g., `--roi 0.0,0.15,0.075,0.65`) which overrides the default settings.
