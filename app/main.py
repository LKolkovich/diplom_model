"""FastAPI application entry-point.

Endpoints
---------
POST /process
    Accepts a ProcessRequest JSON body and enqueues the M0-M6 pipeline as a
    background task.  Returns a task_id immediately.

GET /status/{task_id}
    Returns current progress percentage, module name, and status.

GET /result/{task_id}
    When the task has completed, returns download links to the SRT files.

GET /download/{task_id}/{filename}
    Serves a generated SRT file.

GET /health
    Liveness probe.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.config import get_settings
from app.models import (
    ProcessRequest,
    ProcessResponse,
    ResultResponse,
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
    return {"status": "ok"}


@app.post("/process", response_model=ProcessResponse, status_code=202, tags=["pipeline"])
def start_processing(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
) -> ProcessResponse:
    settings = get_settings()
    task_id = create_task()
    background_tasks.add_task(run_pipeline, task_id, request, settings)
    logger.info("Queued task %s for video: %s", task_id, request.video_path)
    return ProcessResponse(
        task_id=task_id,
        message="Processing started. Poll /status/{task_id} for progress.",
    )


@app.get("/status/{task_id}", response_model=StatusResponse, tags=["pipeline"])
def get_status(task_id: str) -> StatusResponse:
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return StatusResponse(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        current_module=task.current_module,
        error=task.error,
    )


@app.get("/result/{task_id}", response_model=ResultResponse, tags=["pipeline"])
def get_result(task_id: str) -> ResultResponse:
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    if task.status == TaskStatus.running or task.status == TaskStatus.pending:
        raise HTTPException(
            status_code=202,
            detail=f"Task is still {task.status.value}. Progress: {task.progress}%",
        )
    if task.status == TaskStatus.failed:
        raise HTTPException(status_code=500, detail=task.error or "Pipeline failed")

    download_links = {
        label: f"/download/{task_id}/{Path(path).name}"
        for label, path in task.result_files.items()
    }
    return ResultResponse(task_id=task_id, status=task.status, files=download_links)


@app.get("/download/{task_id}/{filename}", tags=["pipeline"])
def download_file(task_id: str, filename: str) -> FileResponse:
    settings = get_settings()
    file_path = settings.output_dir / task_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=str(file_path),
        media_type="text/plain",
        filename=filename,
    )
