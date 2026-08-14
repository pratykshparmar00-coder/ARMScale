# Runtime Decision

## Initial Choice: `llama.cpp`

We have chosen `llama.cpp` (via `llama-cpp-python`) as the initial AI inference runtime for ARMScale.

### Rationale

1. **CPU Inference Optimization**: `llama.cpp` is heavily optimized for CPU execution, which matches the reality of most cost-effective Arm64 cloud environments.
2. **GGUF Support**: The GGUF format allows us to use highly optimized quantized models (e.g., Q4_K_M, INT8), saving memory and improving throughput compared to unquantized models.
3. **Portability**: It works natively across x86_64 and ARM64 without heavy dependencies like PyTorch or CUDA, which simplifies deployment to generic cloud VMs.
4. **Benchmarking Visibility**: The engine provides clear token timing and generation statistics that are crucial for the ARMScale optimization goals.

### Future Alternatives

This is the *initial* runtime choice and may be replaced or supplemented after Arm64 benchmarking. We have built an abstraction layer (`InferenceEngine`) to allow adding:
- **ONNX Runtime**: Excellent for specific Arm optimizations using native acceleration.
- **ExecuTorch**: Meta's edge/mobile runtime, which may have interesting cloud deployment characteristics.
- **LiteRT**: Another highly optimized edge/CPU engine.
