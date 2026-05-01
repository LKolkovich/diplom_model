import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

from app.config import get_settings
from app.models import ProcessConfig, CVDetection
from app.orchestrator import run_pipeline, create_task, get_task

logger = logging.getLogger("app.cli")


def parse_roi(roi_str: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not roi_str:
        return None
    try:
        parts = [float(x) for x in roi_str.split(",")]
        if len(parts) != 4:
            raise ValueError("ROI must be 4 comma-separated floats")
        return (parts[0], parts[1], parts[2], parts[3])
    except ValueError as e:
        logger.error(f"Invalid ROI format: {e}")
        sys.exit(1)


def get_roi_for_mode(
    video_mode: str,
    explicit_roi: Optional[str],
    settings,
) -> Optional[Tuple[float, float, float, float]]:
    if video_mode == "user_crop":
        return None

    if explicit_roi:
        return parse_roi(explicit_roi)

    return settings.roi_as_tuple()


def main():
    parser = argparse.ArgumentParser(description="Valorant Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    full_parser = subparsers.add_parser("run_pipeline", help="Run the complete M1-M6 pipeline")
    full_parser.add_argument("video_path", help="Path to input video")
    full_parser.add_argument("--roi", help="Explicit ROI (x1,y1,x2,y2). Overrides settings if video-mode is full_frame.")
    full_parser.add_argument("--video-mode", choices=["full_frame", "user_crop"], default="full_frame", help="Video mode (default: full_frame)")
    full_parser.add_argument("--hf-token", help="Hugging Face token for WhisperX")
    full_parser.add_argument("--min-speakers", type=int, help="Min speakers")
    full_parser.add_argument("--max-speakers", type=int, help="Max speakers")
    full_parser.add_argument("--language", help="Language code")
    full_parser.add_argument("--initial-prompt", help="Initial prompt for Whisper")
    full_parser.add_argument("--output-dir", help="Output directory")

    full_parser = subparsers.add_parser("run_pipeline", help="Run the complete M1-M6 pipeline")
    full_parser.add_argument("video_path", help="Path to input video")
    full_parser.add_argument("--roi", help="Explicit ROI (x1,y1,x2,y2). Overrides settings if video-mode is full_frame.")
    full_parser.add_argument("--video-mode", choices=["full_frame", "user_crop"], default="full_frame", help="Video mode (default: full_frame)")
    full_parser.add_argument("--hf-token", help="Hugging Face token for WhisperX")
    full_parser.add_argument("--min-speakers", type=int, help="Min speakers")
    full_parser.add_argument("--max-speakers", type=int, help="Max speakers")
    full_parser.add_argument("--language", help="Language code")
    full_parser.add_argument("--initial-prompt", help="Initial prompt for Whisper")
    full_parser.add_argument("--output-dir", help="Output directory")

    # M3 arguments for full pipeline
    full_parser.add_argument("--no-vad", action="store_true", help="Disable VAD in M3")
    full_parser.add_argument("--vad-onset", type=float, default=0.500, help="VAD onset threshold for M3")
    full_parser.add_argument("--vad-offset", type=float, default=0.363, help="VAD offset threshold for M3")
    full_parser.add_argument("--chunk-size", type=int, default=30, help="VAD chunk size in seconds for M3")
    full_parser.add_argument("--level-audio", action="store_true", help="Enable audio preprocessing before M3")
    full_parser.add_argument("--compression-ratio", type=float, default=2.0, help="Audio compression ratio for M3 preprocessing")
    full_parser.add_argument("--compression-threshold", type=float, default=-20.0, help="Audio compression threshold in dBFS for M3 preprocessing")
    full_parser.add_argument("--save-preprocessed-audio", help="Path to save preprocessed WAV for M3 debugging")

    # M4 arguments for full pipeline
    full_parser.add_argument("--fps", type=int, help="Target FPS for M2/M4 inside full pipeline")
    full_parser.add_argument("--debug-frames", action="store_true", help="Enable debug frame output with overlays in M4")
    full_parser.add_argument("--debug-dir", help="Directory for debug overlays in M4")
    full_parser.add_argument("--templates-dir", help="Directory for agent templates in M4")
    full_parser.add_argument("--agents-list", help="Comma-separated list of agent names for M4, e.g. 'skye,clove,sage,killjoy'")

    m2_parser = subparsers.add_parser("run_m2", help="Run only M2: Frame Extraction")
    m2_parser.add_argument("video_path", help="Path to input video")
    m2_parser.add_argument("output_dir", help="Directory to save frames")
    m2_parser.add_argument("--roi", help="Explicit ROI (x1,y1,x2,y2). Used only if video-mode is full_frame.")
    m2_parser.add_argument("--video-mode", choices=["full_frame", "user_crop"], default="full_frame", help="Video mode (default: full_frame)")
    m2_parser.add_argument("--fps", type=int, default=5, help="Target FPS")

    m3_parser = subparsers.add_parser("run_m3", help="Run only M3: WhisperX ASR")
    m3_parser.add_argument("audio_path", help="Path to input video or 16 kHz mono WAV")
    m3_parser.add_argument("output_json", help="Path to output ASR JSON")
    m3_parser.add_argument("--hf-token", help="Hugging Face token for WhisperX")
    m3_parser.add_argument("--min-speakers", type=int, help="Min speakers")
    m3_parser.add_argument("--max-speakers", type=int, help="Max speakers")
    m3_parser.add_argument("--language", default="ru", help="Language code (default: ru)")
    m3_parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD in WhisperX (pass vad_filter=False, no vad_options)",
    )
    m3_parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Начальный промпт для Whisper (словарь терминов)"
    )
    m3_parser.add_argument(
        "--vad-onset", type=float, default=0.500,
        help="Порог начала речи для VAD (default: 0.5)"
    )
    m3_parser.add_argument(
        "--vad-offset", type=float, default=0.363,
        help="Порог окончания речи для VAD (default: 0.363)"
    )
    m3_parser.add_argument(
        "--chunk-size", type=int, default=30,
        help="Размер чанка для VAD в секундах (default: 30)"
    )
    m3_parser.add_argument(
        "--level-audio",
        action="store_true",
        help=(
            "Применить предобработку аудио (компрессия + нормализация громкости). "
            "Полезно при неравномерной громкости спикеров: тихие/спокойные vs громкие/эмоциональные."
        ),
    )
    m3_parser.add_argument(
        "--compression-ratio",
        type=float,
        default=2.0,
        help="Степень компрессии динамики (default: 2.0). Рекомендуется 2.0–3.0 для речи.",
    )
    m3_parser.add_argument(
        "--compression-threshold",
        type=float,
        default=-20.0,
        help="Порог компрессии в dBFS (default: -20.0).",
    )
    m3_parser.add_argument(
        "--save-preprocessed-audio",
        help="Path to save preprocessed WAV for debugging (works only with --level-audio)",
    )

    m4_parser = subparsers.add_parser("run_m4", help="Run only M4: CV Portrait Detection")
    m4_parser.add_argument("video_path", help="Path to input video")
    m4_parser.add_argument("output_json", help="Path to output detections JSON")
    m4_parser.add_argument("--roi", help="Explicit ROI (x1,y1,x2,y2). Used only if video-mode is full_frame.")
    m4_parser.add_argument("--video-mode", choices=["full_frame", "user_crop"], default="full_frame", help="Video mode (default: full_frame)")
    m4_parser.add_argument("--fps", type=int, default=5, help="Target FPS")
    m4_parser.add_argument("--debug-frames", action="store_true", help="Enable debug frame output with overlays")
    m4_parser.add_argument("--debug-dir", help="Directory for debug overlays (default: debug/frames)")
    m4_parser.add_argument("--templates-dir", help="Directory for agent templates")
    m4_parser.add_argument("--agents-list", help="Comma-separated list of agent names to load, e.g. 'skye,clove,sage,killjoy'")

    args = parser.parse_args()

    if args.command == "run_pipeline":
        run_full(args)
    elif args.command == "run_m2":
        run_m2(args)
    elif args.command == "run_m3":
        run_m3(args)
    elif args.command == "run_m4":
        run_m4(args)
    else:
        parser.print_help()


def run_full(args):
    settings = get_settings()

    update_data = {}
    if args.hf_token:
        update_data["hf_token"] = args.hf_token
    if args.output_dir:
        update_data["output_dir"] = Path(args.output_dir)
    if args.roi:
        update_data["voicechat_roi"] = args.roi

    # M4 overrides
    if args.fps is not None:
        update_data["frame_rate"] = args.fps
    if args.templates_dir:
        update_data["templates_dir"] = Path(args.templates_dir)
    if args.agents_list:
        update_data["cv_agents_list"] = args.agents_list
    if args.debug_frames:
        update_data["debug_frames"] = True
    if args.debug_dir:
        update_data["debug_frames_dir"] = Path(args.debug_dir)

    # M3 overrides
    if args.level_audio:
        update_data["m3_level_audio"] = True
    if args.no_vad:
        update_data["m3_no_vad"] = True
    update_data["m3_vad_onset"] = args.vad_onset
    update_data["m3_vad_offset"] = args.vad_offset
    update_data["m3_chunk_size"] = args.chunk_size
    update_data["m3_compression_ratio"] = args.compression_ratio
    update_data["m3_compression_threshold"] = args.compression_threshold
    if args.save_preprocessed_audio:
        update_data["m3_save_preprocessed_audio"] = Path(args.save_preprocessed_audio)

    if update_data:
        settings = settings.model_copy(update=update_data)

    config = ProcessConfig(
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        language=args.language,
        initial_prompt=args.initial_prompt,
        video_mode=args.video_mode,
    )

    task_id = create_task()
    logger.info("Starting full pipeline. Task ID: %s", task_id)

    run_pipeline(task_id, args.video_path, config, settings)

    task = get_task(task_id)
    if task and task.status == "completed":
        logger.info("Pipeline completed successfully.")
        logger.info("Results: %s", task.result_files)
    else:
        error_msg = task.error if task else "Unknown error"
        logger.error("Pipeline failed: %s", error_msg)
        sys.exit(1)


def run_m2(args):
    from app.services.m2_frame_extractor import extract_frames_to_disk

    settings = get_settings()
    roi = get_roi_for_mode(args.video_mode, args.roi, settings)

    logger.info(
        "Extracting frames from %s to %s (mode=%s, ROI=%s)...",
        args.video_path,
        args.output_dir,
        args.video_mode,
        roi,
    )
    paths = extract_frames_to_disk(args.video_path, args.output_dir, roi, args.fps)
    logger.info("Extracted %d frames.", len(paths))


def run_m4(args):
    from app.services.m4_cv_mic_detector import detect_speakers

    settings = get_settings()

    update_data = {}
    if args.agents_list:
        update_data["cv_agents_list"] = args.agents_list
    if update_data:
        settings = settings.model_copy(update=update_data)

    roi = get_roi_for_mode(args.video_mode, args.roi, settings)
    templates_dir = args.templates_dir or settings.templates_dir

    logger.info(
        "Running CV portrait detection on %s (mode=%s, ROI=%s, agents=%s)...",
        args.video_path,
        args.video_mode,
        roi,
        settings.cv_agents_list or "ALL",
    )

    cv_detections, total_frames = detect_speakers(
        video_path=args.video_path,
        roi=roi,
        target_fps=args.fps,
        agent_templates_dir=templates_dir,
        settings=settings,
        debug_frames=args.debug_frames,
        debug_frames_dir=args.debug_dir if args.debug_dir else "debug/frames",
    )

    logger.info("Detected %d speaking events across %d frames.", len(cv_detections), total_frames)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([d.model_dump() for d in cv_detections], f, indent=2, ensure_ascii=False)

    logger.info("Detections saved to %s", output_path)

def run_m3(args):
    import subprocess
    import tempfile

    from app.services.m3_whisperx_asr import transcribe

    settings = get_settings()
    hf_token = args.hf_token or getattr(settings, "hf_token", "")

    audio_path = Path(args.audio_path)
    tmp_wav_path: Optional[Path] = None
    tmp_leveled_path: Optional[Path] = None
    persistent_leveled_path: Optional[Path] = None

    if audio_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        logger.info("M3 – input is video, converting to 16kHz mono WAV via ffmpeg...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = Path(tmp_wav.name)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(tmp_wav_path),
            ],
            check=True,
        )
        audio_path = tmp_wav_path
        logger.info("M3 – converted to: %s", audio_path)

    try:
        if getattr(args, "level_audio", False):
            logger.info(
                "M3 – audio leveling enabled "
                "(compression_ratio=%.1f, threshold=%.1fdB)",
                args.compression_ratio,
                args.compression_threshold,
            )

            save_preprocessed_audio = getattr(args, "save_preprocessed_audio", None)
            if save_preprocessed_audio:
                persistent_leveled_path = Path(save_preprocessed_audio)
                logger.info(
                    "M3 – preprocessed audio will be saved to: %s",
                    persistent_leveled_path,
                )

            processed_audio_path = _preprocess_audio_for_asr(
                input_wav=audio_path,
                compression_ratio=args.compression_ratio,
                compression_threshold_db=args.compression_threshold,
                output_path=persistent_leveled_path,
            )

            if persistent_leveled_path is None:
                tmp_leveled_path = processed_audio_path

            audio_path = processed_audio_path
        else:
            if getattr(args, "save_preprocessed_audio", None):
                logger.warning(
                    "M3 – --save-preprocessed-audio was provided, "
                    "but --level-audio is disabled; nothing will be saved."
                )
            logger.info("M3 – audio leveling disabled (use --level-audio to enable)")

        logger.info(
            "Running WhisperX ASR on %s "
            "(language=%s, min_speakers=%s, max_speakers=%s)...",
            audio_path,
            args.language,
            args.min_speakers,
            args.max_speakers,
        )

        segments = transcribe(
            audio_path=audio_path,
            language=args.language,
            hf_token=hf_token,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            batch_size=getattr(args, "batch_size", 16),
            initial_prompt=getattr(args, "initial_prompt", None),
            vad_filter=not getattr(args, "no_vad", False),
            vad_onset=getattr(args, "vad_onset", 0.500),
            vad_offset=getattr(args, "vad_offset", 0.363),
            chunk_size=getattr(args, "chunk_size", 30),
        )

        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [s.model_dump() for s in segments],
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info("Saved %d ASR segments to %s", len(segments), output_path)

    finally:
        if tmp_leveled_path and tmp_leveled_path.exists():
            tmp_leveled_path.unlink()
            logger.debug("Removed temp leveled WAV: %s", tmp_leveled_path)

        if tmp_wav_path and tmp_wav_path.exists():
            tmp_wav_path.unlink()
            logger.debug("Removed temp WAV: %s", tmp_wav_path)

def _preprocess_audio_for_asr(
        input_wav: Path,
        compression_ratio: float = 2.0,
        compression_threshold_db: float = -20.0,
        attack_ms: float = 50.0,
        release_ms: float = 200.0,
        makeup_gain_db: float = 3.0,
        target_lufs: float = -16.0,
        output_path: Optional[Path] = None,
) -> Path:
    """Применяет лёгкую динамическую компрессию и нормализацию громкости через ffmpeg.

    Полезно когда в одной записи есть спикеры с сильно разной громкостью
    (например, эмоциональные/громкие vs спокойные/тихие). Лёгкий компрессор
    уменьшает разрыв, loudnorm выравнивает итоговый уровень.

    Если output_path не задан, создаётся временный WAV, который вызывающий код
    может удалить после использования. Если output_path задан, результат
    сохраняется в него и может быть использован для отладки.

    Parameters
    ----------
    input_wav:
        Входной 16kHz mono WAV.
    compression_ratio:
        Степень сжатия динамики. 2.0 = мягкая (2:1), 4.0 = заметная (4:1).
    compression_threshold_db:
        Порог (dBFS), выше которого начинается сжатие.
    attack_ms:
        Время атаки компрессора (мс).
    release_ms:
        Время отпускания компрессора (мс).
    makeup_gain_db:
        Компенсирующее усиление после компрессии (dB).
    target_lufs:
        Целевой уровень громкости по EBU R128 (LUFS).
    output_path:
        Куда сохранить предобработанный WAV. Если None — создаётся временный файл.

    Returns
    -------
    Path
        Путь к обработанному WAV файлу.
    """
    import subprocess
    import tempfile

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix="_leveled.wav", delete=False)
        tmp.close()
        output_path = Path(tmp.name)
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    af_filter = (
        f"acompressor="
        f"threshold={compression_threshold_db}dB:"
        f"ratio={compression_ratio}:"
        f"attack={attack_ms}:"
        f"release={release_ms}:"
        f"makeup={makeup_gain_db}dB,"
        f"loudnorm=I={target_lufs}:LRA=11:TP=-1.5"
    )

    logger.info(
        "Audio preprocessing – compressor: threshold=%.1fdB, ratio=%.1f, "
        "attack=%.0fms, release=%.0fms, makeup=%.1fdB; loudnorm: I=%.1f LUFS",
        compression_threshold_db,
        compression_ratio,
        attack_ms,
        release_ms,
        makeup_gain_db,
        target_lufs,
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-af",
            af_filter,
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ],
        check=True,
    )

    logger.info("Audio preprocessing – done, output: %s", output_path)
    return output_path

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
