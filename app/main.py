"""FastAPI application entry-point.

Endpoints
---------
POST /tasks
    Accepts a video file and an optional JSON config via multipart/form-data.
    Enqueues the M1-M6 pipeline as a background task. Returns a task_id.

GET /tasks/{task_id}
    Returns current status, progress, stage, and message.

GET /tasks/{task_id}/result
    When the task has completed, returns a ZIP file containing the SRT files
    and metadata.json.

GET /health
    Liveness probe.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import get_settings
from app.models import (
    ProcessConfig,
    ProcessResponse,
    StatusResponse,
    TaskStatus,
)
from app.orchestrator import create_task, get_task, run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Valorant Subtitle Generation API",
    description=(
        "Transcribes and diarises Valorant in-game voice chat recordings using "
        "WhisperX (ASR) and a CV-based mic-activity detector (pHash + template matching)."
    ),
    version="1.0.0",
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "subtitle-generator",
    }


@app.post("/tasks", response_model=ProcessResponse, status_code=202, tags=["pipeline"])
async def create_task_endpoint(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    config: str = Form("{}"),
) -> ProcessResponse:
    settings = get_settings()
    task_id = create_task()

    # Save uploaded file
    task_dir = settings.output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    video_path = task_dir / video.filename
    with video_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    try:
        config_dict = json.loads(config)
        process_config = ProcessConfig(**config_dict)
    except Exception as e:
        logger.error("Invalid config JSON: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {e}")

    background_tasks.add_task(run_pipeline, task_id, str(video_path), process_config, settings)
    logger.info("Queued task %s for video: %s", task_id, video.filename)
    return ProcessResponse(
        task_id=task_id,
        message="Task created. Poll /tasks/{task_id} for progress.",
    )


@app.get("/tasks/{task_id}", response_model=StatusResponse, tags=["pipeline"])
def get_task_status(task_id: str) -> StatusResponse:
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return StatusResponse(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        stage=task.stage,
        message=task.message,
        current_module=task.current_module,
        error=task.error,
    )


@app.get("/tasks/{task_id}/result", tags=["pipeline"])
def get_task_result(task_id: str) -> FileResponse:
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    if task.status != TaskStatus.completed:
        raise HTTPException(
            status_code=400,
            detail=f"Task is {task.status.value}. Result only available when completed.",
        )

    zip_path = task.result_files.get("zip")
    if not zip_path or not Path(zip_path).exists():
        raise HTTPException(status_code=404, detail="Result ZIP not found")

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"{task_id}_results.zip",
    )
