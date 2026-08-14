# ARMScale Optimization Engine

The ARMScale Optimization Engine is responsible for executing hardware-aware hyperparameter searches across AI inference engines.

## Optimization Dimensions
In this phase, we support **Thread Optimization**. We dynamically scan the available CPU logic (physical vs logical cores) and generate thread candidates (e.g., 1, 2, 4, 6, 8, 12). 
*(Note: Quantization, Context, and Batching optimization are deferred for later phases to maintain strict scientific controls).*

## Benchmark Methodology
- The optimizer uses the exact same `BenchmarkEngine` built in Phase C.
- Warmups: 2
- Measured runs: 5
- Model and Prompts remain fixed.
- After every configuration change, the engine is properly recycled to apply the new hardware bindings.

## Objective Functions
1. **SPEED**: Emphasizes the lowest mean latency (`mean_latency_ms`).
2. **THROUGHPUT**: Emphasizes the highest generation speed (`tokens_per_second`).
3. **MEMORY**: Minimizes RAM usage (`memory_mb`).
4. **BALANCED**: A normalized scoring function equally weighting all three metrics:
   `score = (normalized_latency + normalized_tps + normalized_memory) / 3.0`

## Pareto Analysis
A configuration is added to the Pareto Frontier if it is **non-dominated**. A configuration A dominates B if A is no worse than B in all metrics (latency, throughput) and strictly better in at least one metric. We return all non-dominated configurations so the user can visualize the true trade-offs rather than forcing a single "winner".

## Limitations
- Thread optimizations on some backends may have non-linear scaling due to memory bandwidth limits. 
- Memory profiling in Python is somewhat loose. Native memory profiling will be needed for exact memory optimization on cloud servers.
