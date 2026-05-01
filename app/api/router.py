from __future__ import annotations

import asyncio
import io
import logging
import shutil
import time
import zipfile
from pathlib import Path
from typing import Literal, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.dependencies import get_api_settings
from app.api.schemas import (
    HealthResponse,
    ReadyResponse,
    ResultErrorResponse,
    ResultPendingResponse,
    StatusResponse,
    SubmitResponse,
)
from app.config import Settings
from app.models import ProcessConfig, TaskStatus
from app.orchestrator import create_task, get_task, run_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tasks"])
_pipeline_lock = asyncio.Lock()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse, tags=["ops"])
async def ready() -> ReadyResponse:
    return ReadyResponse(status="ok", pipeline_busy=_pipeline_lock.locked())


def _safe_suffix(filename: Optional[str]) -> str:
    if not filename:
        return ".mp4"
    suffix = Path(filename).suffix
    return suffix if suffix else ".mp4"


async def _save_upload(task_id: str, video: UploadFile, settings: Settings) -> str:
    suffix = _safe_suffix(video.filename)
    task_dir = Path(settings.tmp_dir) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    dst_path = task_dir / f"input{suffix}"

    total_bytes = 0
    started = time.perf_counter()

    try:
        with dst_path.open("wb") as out_file:
            while chunk := await video.read(1024 * 1024):
                out_file.write(chunk)
                total_bytes += len(chunk)
    except Exception as exc:
        logger.exception("task_id=%s | Failed to save upload", task_id)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file") from exc
    finally:
        await video.close()

    logger.info(
        "task_id=%s | Upload saved | path=%s size_bytes=%d elapsed_ms=%.1f",
        task_id,
        dst_path,
        total_bytes,
        (time.perf_counter() - started) * 1000,
        )
    return str(dst_path)


def _parse_agents_list(agents_list: Optional[str]) -> Optional[list[str]]:
    if agents_list is None:
        return None
    values = [x.strip() for x in agents_list.split(",") if x.strip()]
    return values or None


def _build_process_config(
        settings: Settings,
        language: Optional[str],
        min_speakers: Optional[int],
        max_speakers: Optional[int],
        initial_prompt: Optional[str],
        video_mode: Literal["full_frame", "user_crop"],
        roi: Optional[str],
        level_audio: bool,
        compression_ratio: float,
        compression_threshold: float,
        fps: int,
        agents_list: Optional[str],
) -> ProcessConfig:
    data = {
        "language": language if language is not None else getattr(settings, "language", None),
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "initial_prompt": initial_prompt,
        "video_mode": video_mode,
        "roi": roi if roi is not None else getattr(settings, "roi", None),
        "level_audio": level_audio,
        "compression_ratio": compression_ratio,
        "compression_threshold": compression_threshold,
        "fps": fps,
        "agents_list": _parse_agents_list(agents_list),
    }
    data = {k: v for k, v in data.items() if v is not None}
    return ProcessConfig(**data)


def _build_runtime_settings(settings: Settings, hf_token: Optional[str], roi: Optional[str]) -> Settings:
    update = {}
    if hf_token is not None:
        update["hf_token"] = hf_token
    if roi is not None and hasattr(settings, "roi"):
        update["roi"] = roi
    return settings.model_copy(update=update) if update else settings


async def _run_pipeline_serialized(
        task_id: str,
        video_path: str,
        config: ProcessConfig,
        settings: Settings,
) -> None:
    logger.info("task_id=%s | Waiting for pipeline lock", task_id)
    async with _pipeline_lock:
        logger.info("task_id=%s | Pipeline lock acquired", task_id)
        started = time.perf_counter()
        try:
            await asyncio.to_thread(run_pipeline, task_id, video_path, config, settings)
            logger.info(
                "task_id=%s | Pipeline finished elapsed_ms=%.1f",
                task_id,
                (time.perf_counter() - started) * 1000,
                )
        except Exception:
            logger.exception("task_id=%s | Pipeline execution crashed", task_id)


def _status_payload(task) -> StatusResponse:
    stage = getattr(task, "stage", "") or ""
    current_module = getattr(task, "current_module", None)
    if current_module and not stage:
        stage = str(current_module)

    return StatusResponse(
        task_id=task.task_id,
        status=task.status,
        progress=getattr(task, "progress", 0),
        stage=stage,
        message=getattr(task, "message", ""),
        current_module=current_module,
        error=getattr(task, "error", None),
    )


def _build_result_zip(task_id: str, result_files: dict[str, str]) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for agent_name, file_path in result_files.items():
            path = Path(file_path)
            if not path.exists() or not path.is_file():
                logger.warning(
                    "task_id=%s | Missing result file | agent=%s path=%s",
                    task_id,
                    agent_name,
                    file_path,
                )
                continue
            zf.write(path, arcname=f"subtitles/{agent_name}.srt")
    buffer.seek(0)
    return buffer


def _cleanup_tmp(task_id: str, settings: Settings) -> None:
    if not getattr(settings, "auto_cleanup", False):
        return
    tmp_path = Path(settings.tmp_dir) / task_id
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
        logger.info("task_id=%s | tmp cleaned: %s", task_id, tmp_path)


@router.post("", status_code=202, response_model=SubmitResponse)
async def submit_task(
        request: Request,
        background_tasks: BackgroundTasks,
        settings: Settings = Depends(get_api_settings),
        video: UploadFile = File(...),
        language: Optional[str] = Form(None),
        min_speakers: Optional[int] = Form(None),
        max_speakers: Optional[int] = Form(None),
        initial_prompt: Optional[str] = Form(None),
        video_mode: Literal["full_frame", "user_crop"] = Form("full_frame"),
        roi: Optional[str] = Form(None),
        level_audio: bool = Form(False),
        compression_ratio: float = Form(2.0),
        compression_threshold: float = Form(-20.0),
        fps: int = Form(2),
        agents_list: Optional[str] = Form(None),
        hf_token: Optional[str] = Form(None),
) -> SubmitResponse:
    task_id = create_task()

    logger.info(
        "task_id=%s | POST /tasks | client=%s filename=%s content_type=%s "
        "language=%r min_speakers=%r max_speakers=%r initial_prompt=%r "
        "video_mode=%r roi=%r level_audio=%r compression_ratio=%r "
        "compression_threshold=%r fps=%r agents_list=%r hf_token_provided=%s",
        task_id,
        request.client.host if request.client else "unknown",
        video.filename,
        video.content_type,
        language,
        min_speakers,
        max_speakers,
        initial_prompt,
        video_mode,
        roi,
        level_audio,
        compression_ratio,
        compression_threshold,
        fps,
        agents_list,
        hf_token is not None,
        )

    video_path = await _save_upload(task_id, video, settings)

    config = _build_process_config(
        settings=settings,
        language=language,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        initial_prompt=initial_prompt,
        video_mode=video_mode,
        roi=roi,
        level_audio=level_audio,
        compression_ratio=compression_ratio,
        compression_threshold=compression_threshold,
        fps=fps,
        agents_list=agents_list,
    )
    runtime_settings = _build_runtime_settings(settings, hf_token=hf_token, roi=roi)

    background_tasks.add_task(
        _run_pipeline_serialized,
        task_id,
        video_path,
        config,
        runtime_settings,
    )

    return SubmitResponse(task_id=task_id, status="pending", message="Task queued")


@router.get("/{task_id}", response_model=StatusResponse)
async def get_task_status(
        task_id: str,
        settings: Settings = Depends(get_api_settings),
) -> StatusResponse:
    logger.info("task_id=%s | GET /tasks/%s", task_id, task_id)
    task = get_task(task_id)
    if task is None:
        logger.warning("task_id=%s | Task not found", task_id)
        raise HTTPException(status_code=404, detail="Task not found")
    return _status_payload(task)


@router.get(
    "/{task_id}/result",
    responses={
        200: {"content": {"application/zip": {}}},
        202: {"model": ResultPendingResponse},
        500: {"model": ResultErrorResponse},
    },
)
async def download_result(
        task_id: str,
        settings: Settings = Depends(get_api_settings),
):
    logger.info("task_id=%s | GET /tasks/%s/result", task_id, task_id)
    task = get_task(task_id)
    if task is None:
        logger.warning("task_id=%s | Task not found on result", task_id)
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status in {TaskStatus.PENDING, TaskStatus.RUNNING}:
        payload = ResultPendingResponse(
            task_id=task.task_id,
            status=task.status,
            progress=getattr(task, "progress", 0),
        )
        return JSONResponse(status_code=202, content=payload.model_dump(mode="json"))

    if task.status == TaskStatus.FAILED:
        payload = ResultErrorResponse(
            task_id=task.task_id,
            status=task.status,
            error=getattr(task, "error", None),
        )
        logger.error("task_id=%s | Failed task result requested", task_id)
        return JSONResponse(status_code=500, content=payload.model_dump(mode="json"))

    result_files = getattr(task, "result_files", None) or {}
    zip_buffer = _build_result_zip(task_id, result_files)

    _cleanup_tmp(task_id, settings)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="result_{task_id}.zip"'
        },
    )