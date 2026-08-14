# Baseline Benchmarking

## Purpose
The baseline benchmark provides a reproducible reference point for AI inference on a given hardware platform. All future optimizations will be measured against this baseline to calculate relative improvements in latency, throughput, memory, and cost.

## Baseline Configuration
- **Model**: `Qwen2.5-0.5B-Instruct`
- **Quantization**: `Q4_K_M`
- **Context Size**: 2048 tokens
- **Max Tokens Generated**: 128 tokens
- **Temperature**: 0.0 (Deterministic)
- **Threads**: Automatically determined based on physical CPU cores.

## Benchmark Execution
- **Warmup Runs**: 2 (to load model weights into memory and initialize caches)
- **Measured Runs**: 5 (to account for variance)

## Baseline Prompt Suite
A small, deterministic prompt suite is used for the baseline.
1. "What is the capital of France?" (Simple factual)
2. "Explain the theory of relativity in one simple paragraph." (Short explanation)
3. "Write a Python function to calculate the Fibonacci sequence." (Coding)
4. "Summarize the plot of Romeo and Juliet in 3 sentences." (Summarization)
5. "List 5 benefits of using Arm64 processors in cloud computing." (Structured response)
