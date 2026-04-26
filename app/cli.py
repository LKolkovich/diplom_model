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

def get_roi_for_mode(video_mode: str, explicit_roi: Optional[str], settings) -> Optional[Tuple[float, float, float, float]]:
    if video_mode == "user_crop":
        return None
    
    if explicit_roi:
        return parse_roi(explicit_roi)
    
    return settings.roi_as_tuple()

def main():
    parser = argparse.ArgumentParser(description="Valorant Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Full pipeline (renamed from 'full' to 'run_pipeline')
    full_parser = subparsers.add_parser("run_pipeline", help="Run the complete M1-M6 pipeline")
    full_parser.add_argument("video_path", help="Path to input video")
    full_parser.add_argument("--roi", help="Explicit ROI (x1,y1,x2,y2). Overrides settings if video-mode is full_frame.")
    full_parser.add_argument("--video-mode", choices=["full_frame", "user_crop"], help="Video mode")
    full_parser.add_argument("--hf-token", help="Hugging Face token for WhisperX")
    full_parser.add_argument("--min-speakers", type=int, help="Min speakers")
    full_parser.add_argument("--max-speakers", type=int, help="Max speakers")
    full_parser.add_argument("--language", help="Language code")
    full_parser.add_argument("--initial-prompt", help="Initial prompt for Whisper")
    full_parser.add_argument("--output-dir", help="Output directory")

    # Run M2 only
    m2_parser = subparsers.add_parser("run_m2", help="Run only M2: Frame Extraction")
    m2_parser.add_argument("video_path", help="Path to input video")
    m2_parser.add_argument("output_dir", help="Directory to save frames")
    m2_parser.add_argument("--roi", help="Explicit ROI (x1,y1,x2,y2). Used only if video-mode is full_frame.")
    m2_parser.add_argument("--video-mode", choices=["full_frame", "user_crop"], default="full_frame", help="Video mode (default: full_frame)")
    m2_parser.add_argument("--fps", type=int, default=5, help="Target FPS")

    # Run M4 only
    m4_parser = subparsers.add_parser("run_m4", help="Run only M4: CV Mic Detection")
    m4_parser.add_argument("video_path", help="Path to input video")
    m4_parser.add_argument("output_json", help="Path to output detections JSON")
    m4_parser.add_argument("--roi", help="Explicit ROI (x1,y1,x2,y2). Used only if video-mode is full_frame.")
    m4_parser.add_argument("--video-mode", choices=["full_frame", "user_crop"], default="full_frame", help="Video mode (default: full_frame)")
    m4_parser.add_argument("--fps", type=int, default=5, help="Target FPS")
    m4_parser.add_argument("--debug-dir", help="Directory for debug overlays")
    m4_parser.add_argument("--templates-dir", help="Directory for agent templates")

    args = parser.parse_args()

    if args.command == "run_pipeline":
        run_full(args)
    elif args.command == "run_m2":
        run_m2(args)
    elif args.command == "run_m4":
        run_m4(args)
    else:
        parser.print_help()

def run_full(args):
    settings = get_settings()
    if args.hf_token:
        settings.hf_token = args.hf_token
    if args.output_dir:
        settings.output_dir = Path(args.output_dir)
    if args.roi:
        settings.voicechat_roi = args.roi
    
    config = ProcessConfig(
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        language=args.language,
        initial_prompt=args.initial_prompt,
        video_mode=args.video_mode
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
    
    logger.info("Extracting frames from %s to %s (mode=%s, ROI=%s)...", 
                args.video_path, args.output_dir, args.video_mode, roi)
    paths = extract_frames_to_disk(args.video_path, args.output_dir, roi, args.fps)
    logger.info("Extracted %d frames.", len(paths))

def run_m4(args):
    from app.services.m4_cv_mic_detector import detect_speakers
    settings = get_settings()
    
    roi = get_roi_for_mode(args.video_mode, args.roi, settings)
    
    templates_dir = args.templates_dir or settings.templates_dir
    
    logger.info("Running CV mic detection on %s (mode=%s, ROI=%s)...", 
                args.video_path, args.video_mode, roi)
    cv_detections, total_frames = detect_speakers(
        video_path=args.video_path,
        roi=roi,
        target_fps=args.fps,
        agent_templates_dir=templates_dir,
        mic_templates_dir=settings.mic_templates_dir,
        settings=settings,
        debug_frames=bool(args.debug_dir),
        debug_frames_dir=args.debug_dir if args.debug_dir else "debug_frames"
    )
    
    logger.info("Detected %d speaking events across %d frames.", len(cv_detections), total_frames)
    
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump([d.model_dump() for d in cv_detections], f, indent=2)
    
    logger.info("Detections saved to %s", output_path)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    main()
