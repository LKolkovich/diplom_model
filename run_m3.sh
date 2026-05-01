docker run --rm --gpus all `
  -v ${PWD}/data:/workspace/data `
  -v ${PWD}/debug:/workspace/debug `
  -v ${PWD}/app/services/m3_whisperx_asr.py:/workspace/app/services/m3_whisperx_asr.py `
  -v ${PWD}/app/cli.py:/workspace/app/cli.py `
  -v ${PWD}/app/models.py:/workspace/app/models.py `
  -v ${PWD}/app/config.py:/workspace/app/config.py `
  -v ${PWD}/app/services/m5_fusion_engine.py:/workspace/app/services/m5_fusion_engine.py `
  --env-file .env `
  valorant-pipeline `
  python -m app.cli run_m3 `
    /workspace/data/video_clip.mp4 `
    /workspace/debug/m3_output_leveled.json `
    --language ru `
    --min-speakers 3 `
    --max-speakers 3 `
    --level-audio `
    --compression-ratio 2.5 `
    --compression-threshold -18.0 `
    --save-preprocessed-audio /workspace/debug/m3_preprocessed.wav