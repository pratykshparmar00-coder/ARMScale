# ARMScale Optimization Engine

The ARMScale Optimization Engine executes hardware-aware hyperparameter searches across AI inference engines.

## Optimization Dimensions
In this phase, we support **CPU Thread Optimization**. We dynamically scan host CPU capabilities (physical vs logical cores) and generate thread candidates (e.g., 1, 2, 4, 6, 8, 12).
*(Note: Quantization, Context, and Batching optimization are deferred for later phases to maintain strict scientific controls).*

## Benchmark Workload & Statistical Methodology
To ensure scientific comparability:
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (`qwen2.5-0.5b-instruct-q4_k_m.gguf`)
- **Quantization**: `Q4_K_M`
- **Prompt Suite**: Standardized 5-prompt suite (identical prompts and execution order)
- **Token Generation Settings**: `max_tokens = 128`, `temperature = 0.0`
- **Token Accounting**: Strictly counts generated completion tokens: `tokens_per_second = completion_tokens / generation_time_s`
- **Warmups**: 2 runs
- **Measured Runs**: 5 runs
- **Statistical Aggregation**: All statistics (mean, median, p95 with linear interpolation, min, max, standard deviation) are calculated directly from the raw measurements of the 5 runs.
- **Engine Reinitialization**: Model is cleanly reloaded for each candidate configuration to bind CPU threads accurately without counting initialization time in generation latency.

## Objective Functions
1. **SPEED**: Emphasizes minimal generation latency.
   `score = (normalized_latency_score * 0.9) + (normalized_throughput_score * 0.1)`
2. **THROUGHPUT**: Emphasizes maximum token throughput.
   `score = (normalized_throughput_score * 0.9) + (normalized_latency_score * 0.1)`
3. **BALANCED**: Equal 50/50 balance between latency and throughput:
   `score = (normalized_latency_score + normalized_throughput_score) / 2.0`
4. **MEMORY**: *Memory optimization is deferred until native/process-level measurement is implemented.* When selected, falls back to BALANCED with a documented note.

Where `normalized_latency_score` inverts the normalized latency (lower is better, yielding 1.0 at minimum latency), and `normalized_throughput_score` yields 1.0 at maximum throughput.

## Mathematical Pareto Analysis
A configuration A dominates configuration B if and only if:
- `A.mean_latency_ms <= B.mean_latency_ms` AND `A.mean_tokens_per_second >= B.mean_tokens_per_second`
- AND (`A.mean_latency_ms < B.mean_latency_ms` OR `A.mean_tokens_per_second > B.mean_tokens_per_second`)

A configuration is Pareto-optimal (non-dominated) if no other tested configuration dominates it. All non-dominated configurations are returned.

## Baseline Comparison Methodology
Improvements vs baseline are calculated with identical mathematical definitions:
- **Latency Improvement (%)**: `((baseline_latency - candidate_latency) / baseline_latency) * 100`
- **Throughput Improvement (%)**: `((candidate_throughput - baseline_throughput) / baseline_throughput) * 100`
- **Memory Improvement**: Marked as `null` / `unavailable` until native memory profiling is integrated.

## Hardware Awareness & Scope
- All benchmarks in this development phase are performed on **x86_64 development hardware**.
- x86_64 measurements must **NEVER** be reported as Arm64 cloud results.
