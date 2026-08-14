# Models Directory

This directory stores the AI models used by ARMScale for inference and benchmarking. 

## Requirements
- All models must be in GGUF format to be compatible with the current `llama.cpp` inference engine.
- Do **not** commit model binaries (`.gguf` files) to version control. They are excluded via `.gitignore`.

## Default Baseline Model
The system uses `qwen2.5-0.5b-instruct-q4_k_m.gguf` for baseline testing.
You can download it using the provided tool:
```bash
python tools/download_model.py
```
