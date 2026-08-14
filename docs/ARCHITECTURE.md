# ARMScale Architecture

ARMScale uses a modular inference gateway architecture designed to optimize performance on Arm64 cloud hardware.

```text
                    Client
                      │
                      ▼
              ┌───────────────┐
              │ ARMScale API  │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │ Request Queue │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │ Model Router  │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │ Inference     │
              │ Runtime       │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │ Benchmark     │
              │ Engine        │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │ Optimization  │
              │ Engine        │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │ Metrics Store │
              └───────────────┘
```

## Components

1. **ARMScale API**: A FastAPI interface exposing `/health`, `/api/model`, and `/api/generate` endpoints.
2. **Inference Engine**: An abstract class `InferenceEngine` that allows dynamic switching of the underlying inference backend. Our initial implementation uses `llama.cpp` via `llama-cpp-python` for CPU inference.
3. **Benchmark Engine**: Measures end-to-end performance metrics, isolating actual inference time from network latency.
4. **Optimization Engine** (Future): Explores the configuration space (threads, quantization, context) to identify Pareto-optimal configurations.
