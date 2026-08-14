# ARMScale Experiments

This guide documents how to execute optimization experiments using the ARMScale CLI and API.

## Running Optimization Experiments

Use the CLI to run hardware-aware sweeps:

```bash
# Optimize for speed (latency minimized)
python tools/optimize.py --objective speed

# Optimize for throughput (tokens/sec maximized)
python tools/optimize.py --objective throughput

# Optimize with a balanced objective (50% latency, 50% throughput)
python tools/optimize.py --objective balanced
```

To test specific thread counts:
```bash
python tools/optimize.py --objective speed --threads 1,2,4,6,8,12
```

## Result Artifacts
Each experiment is assigned a unique `experiment_id` and saved in JSON format under:
`benchmarks/results/optimization/optimization_<timestamp>_<experiment_id>.json`

Individual per-configuration benchmarks are saved under:
`benchmarks/results/benchmark_<timestamp>.json` and `.csv`

All artifacts preserve full statistical granularity:
- `mean_latency_ms`, `median_latency_ms`, `p95_latency_ms`, `min_latency_ms`, `max_latency_ms`, `std_latency_ms`
- `mean_tokens_per_second`, `median_tokens_per_second`, `p95_tokens_per_second`, `min_tokens_per_second`, `max_tokens_per_second`, `std_tokens_per_second`
- `memory_mb`: `null` (with `memory_status: "unavailable"`)
- `pareto_configurations`: Non-dominated configuration set
- `improvement_vs_baseline`: Exact percentage deltas against canonical baseline
