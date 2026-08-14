# ARMScale Optimization Engine

The ARMScale Optimization Engine executes hardware-aware hyperparameter searches across AI inference engines.

## Optimization Dimensions
ARMScale currently supports multi-dimensional search:
1. **CPU Thread Optimization**: Dynamically evaluates candidate thread allocations based on host physical and logical core capacity (e.g., 1, 2, 4, 6, 8, 12).
2. **Context Window Optimization**: Evaluates memory and compute trade-offs across context sizes (e.g., 1024, 2048, 4096 tokens).
3. **Combined Multi-Dimensional Search**: Jointly searches thread and context parameter spaces.
*(Note: Quantization variant optimization and batching are modeled in abstractions and deferred to future phases).*

## Workload Suites
- **`short_generation`**: Standard short-turn queries (~15 input tokens, 128 max completion tokens).
- **`context_stress`**: Document-grounded analysis (~650 input tokens, 128 max completion tokens).

## Benchmark Methodology
To ensure scientific comparability:
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (`qwen2.5-0.5b-instruct-q4_k_m.gguf`)
- **Quantization**: `Q4_K_M`
- **Warmups**: 2 runs
- **Measured Runs**: 5 runs
- **Token Accounting**: Strictly counts generated completion tokens: `tokens_per_second = completion_tokens / generation_time_s`
- **Statistical Aggregation**: All statistics (mean, median, p95 with linear interpolation, min, max, standard deviation) are calculated directly from raw measured runs.

## Objective Functions
1. **SPEED**: Emphasizes minimal generation latency.
   `score = (normalized_latency_score * 0.9) + (normalized_throughput_score * 0.1)`
2. **THROUGHPUT**: Emphasizes maximum token throughput.
   `score = (normalized_throughput_score * 0.9) + (normalized_latency_score * 0.1)`
3. **BALANCED**: Equal 50/50 balance between latency and throughput:
   `score = (normalized_latency_score + normalized_throughput_score) / 2.0`
4. **MEMORY**: *Memory optimization is deferred until native/process-level measurement is implemented.* When selected, falls back to BALANCED with a documented note.

## Mathematical Pareto Analysis
A configuration A dominates configuration B if and only if:
- `A.mean_latency_ms <= B.mean_latency_ms` AND `A.mean_tokens_per_second >= B.mean_tokens_per_second`
- AND (`A.mean_latency_ms < B.mean_latency_ms` OR `A.mean_tokens_per_second > B.mean_tokens_per_second`)

All non-dominated configurations form the empirical Pareto Frontier.

## Platform Independence
The optimization engine queries the `PlatformAdapter` abstraction layer, ensuring identical execution on local development machines, Google Axion Arm64 instances, and generic Arm64 bare metal.
