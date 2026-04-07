"""M0 – Orchestrator.

Coordinates the M1→M6 pipeline for a single processing task.  Progress is
tracked in a shared in-memory task store so that the FastAPI status endpoint
can return live progress percentages.

Task lifecycle
--------------
pending → running → completed | failed

Progress milestones per module
-------------------------------
M1 AudioValidator   10 %
M2 FrameExtractor   25 %
M3 WhisperXASR      55 %
M4 CVMicDetector    70 %
M5 FusionEngine     85 %
M6 SRTExporter      100 %
"""

from __future__ import annotations

import logging
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Dict, Optional

from app.config import Settings
from app.models import ModuleTag, ProcessRequest, TaskState, TaskStatus
from app.services import (
    m1_audio_validator,
    m2_frame_extractor,
    m3_whisperx_asr,
    m4_cv_mic_detector,
    m5_fusion_engine,
    m6_srt_exporter,
)

logger = logging.getLogger(__name__)

_TASK_STORE: Dict[str, TaskState] = {}

PROGRESS_MAP = {
    ModuleTag.m1_validate: 10,
    ModuleTag.m2_frames: 25,
    ModuleTag.m3_asr: 55,
    ModuleTag.m4_cv: 70,
    ModuleTag.m5_fusion: 85,
    ModuleTag.m6_export: 100,
}


def create_task() -> str:
    task_id = uuid.uuid4().hex
    _TASK_STORE[task_id] = TaskState(task_id=task_id)
    return task_id


def get_task(task_id: str) -> Optional[TaskState]:
    return _TASK_STORE.get(task_id)


def _update(task: TaskState, module: ModuleTag) -> None:
    task.current_module = module
    task.progress = PROGRESS_MAP[module]
    task.status = TaskStatus.running


def run_pipeline(task_id: str, request: ProcessRequest, settings: Settings) -> None:
    """Execute the full M1-M6 pipeline in a background thread.

    This function is designed to be called from ``FastAPI.BackgroundTasks`` and
    must not raise – all exceptions are caught and recorded in the task state.
    """
    task = _TASK_STORE.get(task_id)
    if task is None:
        logger.error("run_pipeline: unknown task_id '%s'", task_id)
        return

    task.status = TaskStatus.running
    tmp_dir = settings.output_dir / task_id / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        _update(task, ModuleTag.m1_validate)
        logger.info("[%s] M1 – AudioValidator", task_id)
        audio_source = request.audio_path or request.video_path
        wav_path = m1_audio_validator.validate_and_convert(audio_source, tmp_dir)

        _update(task, ModuleTag.m2_frames)
        logger.info("[%s] M2 – FrameExtractor", task_id)
        roi = settings.roi_as_tuple()

        _update(task, ModuleTag.m3_asr)
        logger.info("[%s] M3 – WhisperXASR", task_id)
        asr_segments = m3_whisperx_asr.transcribe(
            audio_path=wav_path,
            model_name=settings.whisper_model,
            device=settings.device,
            compute_type=settings.compute_type,
            language=request.language,
            hf_token=settings.hf_token,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
        )

        _update(task, ModuleTag.m4_cv)
        logger.info("[%s] M4 – CVMicDetector", task_id)
        cv_detections = m4_cv_mic_detector.detect_speakers(
            video_path=request.video_path,
            roi=roi,
            target_fps=settings.frame_rate,
            templates_dir=settings.templates_dir,
            phash_threshold=settings.phash_threshold,
        )

        _update(task, ModuleTag.m5_fusion)
        logger.info("[%s] M5 – FusionEngine", task_id)
        fused = m5_fusion_engine.fuse(asr_segments, cv_detections)

        _update(task, ModuleTag.m6_export)
        logger.info("[%s] M6 – SRTExporter", task_id)
        output_dir = settings.output_dir / task_id
        result_files = m6_srt_exporter.export(fused, output_dir, task_id)

        task.result_files = result_files
        task.status = TaskStatus.completed
        task.current_module = ModuleTag.done
        logger.info("[%s] pipeline completed. files: %s", task_id, list(result_files.keys()))

    except Exception:
        task.status = TaskStatus.failed
        task.error = traceback.format_exc()
        logger.exception("[%s] pipeline failed", task_id)

    finally:
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
                logger.debug("[%s] cleaned up tmp dir", task_id)
        except OSError as exc:
            logger.warning("[%s] could not clean up tmp dir: %s", task_id, exc)
