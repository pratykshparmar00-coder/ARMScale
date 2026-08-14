# Context Window Optimization & Workload Separation

## Overview
Allocating the context window (`n_ctx`) in large language model inference engines reserves memory for the Key-Value (KV) cache. An oversized context allocation increases memory footprint and can degrade cache locality, while an undersized context truncates prompts or causes out-of-context errors.

ARMScale tests multiple context window sizes (e.g., 1024, 2048, 4096) across differentiated workloads to discover the optimal configuration for specific operational profiles.

---

## Workload Suites

To evaluate context effects scientifically, ARMScale separates benchmarks into two distinct workloads:

### 1. `short_generation` (v1.0)
- **Characteristics**: Short user queries (~15 input tokens).
- **Workload**: 5 standard questions covering factual QA, code generation, summarization, and concept explanation.
- **Generation Budget**: `max_tokens = 128`, `temperature = 0.0`.
- **Purpose**: Measures standard interactive latency where the KV-cache is minimal.

### 2. `context_stress` (v1.0)
- **Characteristics**: Document-grounded queries with a ~500-word passage (~650 input tokens).
- **Workload**: 5 analytical questions requiring parsing and synthesizing the technical passage.
- **Generation Budget**: `max_tokens = 128`, `temperature = 0.0`.
- **Purpose**: Exercises prompt processing (prefill) and memory bandwidth under substantial KV-cache pressure.

---

## Candidate Context Sizes
- **1024 tokens**: Minimal footprint, optimized for compact microservices and short-turn QA.
- **2048 tokens**: Default general-purpose configuration.
- **4096 tokens**: Extended context for multi-document reasoning and retrieval-augmented generation (RAG).

---

## Zero-Fabrication Discovery Rule
ARMScale's optimizer does not assume or simulate improvements. If varying the context size produces negligible latency differences on short workloads, or if a larger context adds slight prefill overhead on long workloads, the actual measured data is recorded and reported without distortion.
