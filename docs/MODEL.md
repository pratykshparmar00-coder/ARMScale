# Model Information

## Selected Model
- **Name**: Qwen2.5-0.5B-Instruct
- **Repository**: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- **Filename**: `qwen2.5-0.5b-instruct-q4_k_m.gguf`
- **Quantization**: `Q4_K_M`
- **License**: Apache 2.0
- **Size**: ~398 MB
- **Source URL**: `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf`

## Rationale
This model is selected for the baseline implementation because:
1. It is extremely lightweight, making rapid iteration and benchmarking fast.
2. It is instruction-tuned and capable of following the baseline prompt suite.
3. The Apache 2.0 license is fully compatible with open-source hackathon requirements.
4. The `Q4_K_M` quantization provides the best balance of size, inference speed, and output quality for CPU inference tests on both x86_64 development and Arm64 target environments.
