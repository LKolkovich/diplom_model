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
import zipfile
from pathlib import Path
from typing import Dict, Optional

import soundfile as sf

from app.config import Settings
from app.models import ModuleTag, ProcessConfig, TaskState, TaskStatus
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

STAGES = {
    "VALIDATING": {"progress": 10, "message": "Validating audio/video inputs"},
    "EXTRACTING_FRAMES": {"progress": 25, "message": "Extracting frames for CV analysis"},
    "TRANSCRIBING": {"progress": 55, "message": "Transcribing audio with WhisperX"},
    "DETECTING_MIC": {"progress": 70, "message": "Detecting mic activity via CV"},
    "FUSING": {"progress": 85, "message": "Fusing ASR and CV results"},
    "EXPORTING": {"progress": 95, "message": "Exporting results to SRT/ZIP"},
}


def create_task() -> str:
    task_id = uuid.uuid4().hex
    _TASK_STORE[task_id] = TaskState(task_id=task_id)
    return task_id


def get_task(task_id: str) -> Optional[TaskState]:
    return _TASK_STORE.get(task_id)


def delete_task(task_id: str) -> None:
    _TASK_STORE.pop(task_id, None)


def _update(
    task: TaskState,
    stage: str,
    module: Optional[ModuleTag] = None,
    message: Optional[str] = None,
) -> None:
    task.stage = stage
    if stage in STAGES:
        task.progress = STAGES[stage]["progress"]
        task.message = message or STAGES[stage]["message"]
    if module:
        task.current_module = module
    task.status = TaskStatus.running


def run_pipeline(
    task_id: str,
    video_path: str,
    config: ProcessConfig,
    settings: Settings,
    portraits_override: Optional[str] = None,
) -> None:
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
        _update(task, "VALIDATING", ModuleTag.m1_validate)
        logger.info("[%s] M1 – AudioValidator", task_id)
        wav_path = m1_audio_validator.validate_and_convert(video_path, tmp_dir)

        _update(task, "EXTRACTING_FRAMES", ModuleTag.m2_frames)
        logger.info("[%s] M2 – FrameExtractor", task_id)
        roi = settings.roi_as_tuple()

        _update(task, "TRANSCRIBING", ModuleTag.m3_asr)
        logger.info("[%s] M3 – WhisperXASR", task_id)
        asr_segments = m3_whisperx_asr.transcribe(
            audio_path=wav_path,
            model_name=settings.whisper_model,
            device=settings.device,
            compute_type=settings.compute_type,
            language=config.language,
            hf_token=settings.hf_token,
            min_speakers=config.min_speakers,
            max_speakers=config.max_speakers,
            initial_prompt=config.initial_prompt,
        )

        _update(task, "DETECTING_MIC", ModuleTag.m4_cv)
        logger.info("[%s] M4 – CVMicDetector", task_id)
        cv_detections = m4_cv_mic_detector.detect_speakers(
            video_path=video_path,
            roi=roi,
            target_fps=settings.frame_rate,
            templates_dir=portraits_override or settings.templates_dir,
            phash_threshold=settings.phash_threshold,
        )

        _update(task, "FUSING", ModuleTag.m5_fusion)
        logger.info("[%s] M5 – FusionEngine", task_id)
        fused = m5_fusion_engine.fuse(asr_segments, cv_detections)

        _update(task, "EXPORTING", ModuleTag.m6_export)
        logger.info("[%s] M6 – SRTExporter", task_id)
        output_dir = settings.output_dir / task_id

        audio_info = sf.info(str(wav_path))
        audio_duration = audio_info.duration

        result_files = m6_srt_exporter.export(
            fused,
            output_dir,
            task_id,
            audio_duration=audio_duration,
            cv_detections=cv_detections,
        )

        # Package results into a ZIP
        zip_path = output_dir / "results.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path_str in result_files.values():
                file_path = Path(file_path_str)
                zipf.write(file_path, file_path.name)

        task.result_files = {"zip": str(zip_path)}
        task.status = TaskStatus.completed
        task.stage = "COMPLETED"
        task.progress = 100
        task.message = "Processing completed successfully"
        task.current_module = ModuleTag.done
        logger.info("[%s] pipeline completed. ZIP: %s", task_id, zip_path)

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
