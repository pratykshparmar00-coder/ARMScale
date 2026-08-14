# Baseline Benchmarking

## Purpose
The baseline benchmark establishes the canonical reference point for AI inference on a given hardware environment. All candidate optimization sweeps are evaluated against this recorded reference using identical mathematical metrics.

## Reference Baseline Configuration (Phase C)
- **Model**: `Qwen2.5-0.5B-Instruct-GGUF` (`qwen2.5-0.5b-instruct-q4_k_m.gguf`)
- **Quantization**: `Q4_K_M`
- **Context Size**: 2048 tokens
- **Max Tokens Generated**: 128 tokens
- **Temperature**: 0.0 (Deterministic)
- **Batch Size**: 1
- **Threads**: 4 (Default baseline configuration)
- **Environment**: DEVELOPMENT MACHINE — x86_64 (AMD 6 physical / 12 logical cores, 15.42 GB RAM)

## Canonical Baseline Measurements (Phase C)
- **Mean Latency**: 2671.42 ms
- **Median Latency**: 3468.52 ms
- **P95 Latency**: 3723.29 ms
- **Mean Throughput**: 32.73 tokens/sec
- **Memory**: `null` (unavailable; native memory profiling deferred)

## Benchmark Execution & Workload Integrity
- **Warmup Runs**: 2 (to eliminate cold-start cache anomalies)
- **Measured Runs**: 5 (fixed prompt suite in deterministic order)
- **Token Accounting**: Strictly measures output `completion_tokens / generation_time_seconds`

## Standard Prompt Suite
1. "What is the capital of France?" (Factual)
2. "Explain the theory of relativity in one simple paragraph." (Explanation)
3. "Write a Python function to calculate the Fibonacci sequence." (Coding)
4. "Summarize the plot of Romeo and Juliet in 3 sentences." (Summarization)
5. "List 5 benefits of using Arm64 processors in cloud computing." (Arm Structured Response)
