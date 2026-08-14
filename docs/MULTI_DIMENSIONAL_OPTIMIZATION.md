# ARMScale Joint Multi-Dimensional Optimization

## Overview
AI inference performance on cloud and edge processors is governed by multi-dimensional parameter interactions rather than isolated variables. ARMScale's Joint Optimization Engine tests combinations of CPU threads and context window allocations simultaneously to find Pareto-optimal configurations and provide goal-oriented recommendations.

---

## Search Space Matrix
The default combined sweep tests 12 configurations:
- **Threads**: `[2, 4, 6, 8]`
- **Context Window (`n_ctx`)**: `[1024, 2048, 4096]`
- **Batch Size**: `1` (deferred for concurrent serving)
- **Quantization**: `Q4_K_M` (standard baseline)

---

## Differentiated Workload Suites
To prevent skewing recommendations, optimizations are benchmarked separately for distinct operational workloads:

1. **`short_generation` (v1.0)**:
   - 5 standard interactive prompts (~15 input tokens, 128 max completion tokens).
   - Characterizes lightweight interactive chatbot and QA latency.

2. **`context_stress` (v1.0)**:
   - Technical document passage (~650 input tokens, 128 max completion tokens).
   - Characterizes heavy prefill compute and KV-cache memory bandwidth pressure.

---

## Recommendation Engine Logic
ARMScale recommends configurations based on verified metrics and chosen operational objectives:

- **Speed Objective**: Selects the configuration with the minimal mean latency (`mean_latency_ms`).
- **Throughput Objective**: Selects the configuration with the highest generation rate (`mean_tokens_per_second`).
- **Balanced Objective**: Selects the configuration with the highest normalized score:
  $$\text{Score} = \frac{\text{Normalized Latency Score} + \text{Normalized Throughput Score}}{2.0}$$

---

## Empirical Pareto Frontier
A candidate configuration $A$ strictly dominates $B$ ($A \succ B$) if:
$$\text{Latency}_A \le \text{Latency}_B \quad \text{and} \quad \text{Throughput}_A \ge \text{Throughput}_B$$
$$\text{with at least one strict inequality.}$$

All configurations where no other configuration strictly dominates are marked `pareto_optimal: true` and surfaced as optimal trade-offs.

---

## Platform Architecture
All experiment results are tagged with the standardized platform metadata schema, ensuring immediate portability between local x86_64 development environments and Google Axion Arm64 cloud infrastructure.
