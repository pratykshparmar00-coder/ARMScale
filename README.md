# ARMScale
**Autonomous Arm64 AI Inference Optimizer**

## Problem
AI inference is expensive and configuration-sensitive. The optimal combination of runtime parameters (threads, batch sizes, quantization formats, and context handling) varies dramatically across hardware architectures. Setting these manually is a trial-and-error process that often leaves cloud resources underutilized.

## Solution
ARMScale automatically benchmarks and optimizes AI inference on Arm64 cloud infrastructure. It profiles the model across various configurations, constructs a Pareto frontier of trade-offs (Latency vs Throughput vs Memory), and recommends the most cost-effective deployment strategy.

## Why Arm64
Arm-based processors in the cloud (such as AWS Graviton, Google Axion, and Microsoft Cobalt) offer significant price-performance advantages. ARMScale is purpose-built to extract maximum efficiency from these environments.

## Architecture
ARMScale uses an abstract inference router, meaning it isn't tied to a single framework.
1. **API Gateway:** Handles inference requests securely.
2. **Inference Engine:** Loads and serves models (currently utilizing `llama.cpp` for CPU inference).
3. **Benchmark Engine:** Rigorously profiles performance using high-resolution monotonic timing.
4. **Optimization Engine (Roadmap):** Automatically searches the configuration space.

## Supported Runtimes
- `llama.cpp` (Current via `llama-cpp-python`)
- (Roadmap: ONNX Runtime, ExecuTorch, LiteRT)

## Supported Models
The architecture supports GGUF format models.
Baseline benchmarking uses: `Qwen2.5-0.5B-Instruct-GGUF` (Q4_K_M).

## Optimization Dimensions
- **Latency** (SPEED)
- **Throughput** (THROUGHPUT)
- **Memory Usage** (MEMORY)
- **Balanced Profile**
- **Estimated Cost** (COST)

## Benchmark Methodology
ARMScale executes deterministic prompt suites with warmup cycles and multiple measured passes, eliminating network jitter by tracking true model inference time from within the engine itself.

## Deployment
ARMScale is designed to be fully containerized for Arm64 cloud environments.
```bash
# Setup
python -m venv .venv
# Activate venv
pip install -r requirements.txt
```

## Arm64 Validation
ARMScale includes environmental awareness. It will clearly distinguish x86_64 development environments from true Arm64 benchmark-eligible infrastructure to ensure measurement integrity.

## Reproducibility
All benchmarks generate timestamped JSON and CSV artifacts detailing precise hardware configurations, runtime settings, and measured percentiles.

## Current Results
**Arm64 benchmark results: PENDING**
