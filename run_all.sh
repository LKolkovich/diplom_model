docker run --rm --gpus all `
  -v ${PWD}/data:/workspace/data `
  -v ${PWD}/templates:/workspace/templates `
  -v ${PWD}/debug:/workspace/debug `
  -v ${PWD}/output:/workspace/output `
  -v ${PWD}/app/services/m3_whisperx_asr.py:/workspace/app/services/m3_whisperx_asr.py `
  -v ${PWD}/app/services/m4_cv_mic_detector.py:/workspace/app/services/m4_cv_mic_detector.py `
  -v ${PWD}/app/services/m5_fusion_engine.py:/workspace/app/services/m5_fusion_engine.py `
  -v ${PWD}/app/cli.py:/workspace/app/cli.py `
  -v ${PWD}/app/models.py:/workspace/app/models.py `
  -v ${PWD}/app/config.py:/workspace/app/config.py `
  --env-file .env `
  valorant-pipeline `
  python -m app.cli run_pipeline `
    /workspace/data/video_clip.mp4 `
    --language ru `
    --min-speakers 3 `
    --max-speakers 3 `
    --level-audio `
    --compression-ratio 2.5 `
    --compression-threshold -18.0 `
    --save-preprocessed-audio /workspace/debug/m3_preprocessed.wav `
    --fps 2 `
    --debug-frames `
    --debug-dir /workspace/debug/frames `
    --templates-dir /workspace/templates/agents `
    --agents-list "clove,sage,killjoy" `
    --output-dir /workspace/output