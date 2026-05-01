docker run --rm --gpus all `
  -v ${PWD}/data:/workspace/data `
  -v ${PWD}/debug:/workspace/debug `
  -v ${PWD}/app/services/m3_whisperx_asr.py:/workspace/app/services/m3_whisperx_asr.py `
  -v ${PWD}/app/services/m5_fusion_engine.py:/workspace/app/services/m5_fusion_engine.py `
  -v ${PWD}/app/models.py:/workspace/app/models.py `
  -v ${PWD}/app/config.py:/workspace/app/config.py `
  -v ${PWD}/app/cli.py:/workspace/app/cli.py `
  --env-file .env `
  valorant-pipeline `
  python -m app.cli run_m5 `
    /workspace/debug/m3_output_leveled.json `
    /workspace/debug/m4_output.json `
    /workspace/debug/m5_fused.json