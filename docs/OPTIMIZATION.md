# ARMScale Optimization Engine

The ARMScale Optimization Engine executes hardware-aware hyperparameter searches across AI inference engines.

## Optimization Dimensions
ARMScale supports comprehensive multi-dimensional search:
1. **CPU Thread Optimization**: Dynamically evaluates candidate thread allocations based on host physical and logical core capacity (`2, 4, 6, 8, 12`).
2. **Context Window Optimization**: Evaluates memory and compute trade-offs across context sizes (`1024, 2048, 4096` tokens).
3. **Quantization Format Optimization**: Evaluates numerical precision formats (`Q4_K_M`, `Q5_K_M`, `Q8_0`).
4. **Autonomous Global Optimization**: Jointly searches the 36-configuration Cartesian product (3 Quantizations $\times$ 4 Threads $\times$ 3 Context sizes).

## Workload Suites
- **`short_generation`**: Standard short-turn queries (~15 input tokens, 128 max completion tokens).
- **`context_stress`**: Document-grounded analysis (~650 input tokens, 128 max completion tokens).

## Benchmark Methodology
To ensure scientific comparability:
- **Model Family**: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- **Supported Quantizations**: `Q4_K_M` (468.6MB), `Q5_K_M` (498.0MB), `Q8_0` (644.4MB)
- **Warmups**: 2 runs
- **Measured Runs**: 5 runs
- **Token Accounting**: Strictly counts generated completion tokens: `tokens_per_second = completion_tokens / generation_time_s`
- **Statistical Aggregation**: All statistics (mean, median, p95 with linear interpolation, min, max, standard deviation) are calculated directly from raw measured runs.

## Objective Functions
1. **SPEED**: Emphasizes minimal generation latency:
   $$\text{Score} = (\text{Latency Score} \times 0.9) + (\text{Throughput Score} \times 0.1)$$
2. **THROUGHPUT**: Emphasizes maximum token throughput:
   $$\text{Score} = (\text{Throughput Score} \times 0.9) + (\text{Latency Score} \times 0.1)$$
3. **SIZE**: Emphasizes minimal model file footprint:
   $$\text{Score} = (\text{Size Score} \times 0.8) + (\text{Latency Score} \times 0.1) + (\text{Throughput Score} \times 0.1)$$
4. **BALANCED**: Equal 50/50 balance between latency and throughput:
   $$\text{Score} = \frac{\text{Latency Score} + \text{Throughput Score}}{2.0}$$
5. **MEMORY**: When memory isolation is unavailable, falls back to BALANCED with a documented note.

## Mathematical 3D Pareto Analysis
A configuration $A$ strictly dominates $B$ ($A \succ B$) if and only if:
- $\text{Latency}_A \le \text{Latency}_B$ AND $\text{Throughput}_A \ge \text{Throughput}_B$ AND $\text{Size}_A \le \text{Size}_B$
- AND at least one strict inequality holds.

All non-dominated configurations form the empirical Pareto Frontier.

## Platform Independence
The optimization engine queries the `PlatformAdapter` abstraction layer, ensuring identical execution on local development machines, Google Axion Arm64 instances, and generic Arm64 bare metal.
