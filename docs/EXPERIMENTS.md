# ARMScale Experiments

This guide explains how to run optimization experiments using the ARMScale CLI.

## Running an Optimization Experiment

You can trigger an optimization sweep via the CLI tool:

```bash
# Optimize for latency
python tools/optimize.py --objective speed

# Optimize for throughput
python tools/optimize.py --objective throughput

# Optimize for balanced performance
python tools/optimize.py --objective balanced
```

You can optionally specify exactly which thread counts to test to constrain the search space:
```bash
python tools/optimize.py --objective speed --threads 1,2,4,6
```

## Results Artifacts

Every experiment is assigned a unique `experiment_id` and saved to:
`benchmarks/results/optimization/optimization_<timestamp>_<experiment_id>.json`

These JSON files contain the full baseline reference, configurations tested, Pareto frontier analysis, and the overall "winner" configuration based on your requested objective.
