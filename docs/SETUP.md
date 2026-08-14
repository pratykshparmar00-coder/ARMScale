# ARMScale Setup Guide

## Requirements
- Python 3.12+ (isolated virtual environment recommended)
- `llama-cpp-python` compatible hardware

## Local Installation (Development)

1. Create virtual environment:
```bash
python -m venv .venv
```

2. Activate virtual environment:
```bash
# Windows
.venv\Scripts\Activate.ps1
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. If `llama-cpp-python` fails to compile on Windows, you can install the prebuilt CPU wheel:
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```
