# ARMScale Autonomous Global Multi-Dimensional Optimization

## 1. Executive Summary
ARMScale Phase H integrates all architectural tuning dimensions into a unified autonomous global optimization engine. The search space evaluates the complete Cartesian product across:

$$\text{Search Space} = \text{Quantizations (3)} \times \text{Threads (4)} \times \text{Context Sizes (3)} = \mathbf{36\text{ Configurations}}$$

For every configuration, the engine executes 2 warmups and 5 measured runs, evaluating **252 inference runs per workload** (**504 total inference runs** across both suites).

---

## 2. Search Space Matrix
- **Quantization Formats**: `["Q4_K_M", "Q5_K_M", "Q8_0"]`
- **CPU Threads**: `[2, 4, 6, 8]`
- **Context Allocation (`n_ctx`)**: `[1024, 2048, 4096]`
- **Batch Size**: `1`
- **Configuration ID Format**: `cfg_<QUANT>_T<THREADS>_C<CONTEXT>` (e.g. `cfg_Q4_K_M_T6_C4096`)

---

## 3. Workload Suites & Methodology
- **`short_generation`**: 5 standard interactive prompts (~15 input tokens, 128 max tokens, temp 0.0). Characterizes interactive latency.
- **`context_stress`**: Document-grounded technical passage (~650 input tokens, 128 max tokens, temp 0.0). Characterizes prefill memory bandwidth and KV-cache pressure.
- **Statistical Fidelity**: All statistics (mean, median, P95 via linear interpolation, min, max, std dev, tokens/sec) are calculated strictly from the 5 measured runs; warmups are completely excluded.

---

## 4. Multi-Objective 3D Pareto Analysis
Pareto dominance is calculated across:
1. **Mean Latency ($ms$)**: Lower is better ($\downarrow$)
2. **Mean Throughput ($tok/s$)**: Higher is better ($\uparrow$)
3. **Model Footprint ($MB$)**: Lower is better ($\downarrow$)

A configuration $A$ strictly dominates $B$ ($A \succ B$) if and only if:
$$(\text{Latency}_A \le \text{Latency}_B) \land (\text{Throughput}_A \ge \text{Throughput}_B) \land (\text{Size}_A \le \text{Size}_B)$$
$$\text{with at least one strict inequality.}$$

---

## 5. Measured Global Optimization Results

### A. Short Generation Workload (`short_generation`, 36 Configs)
*Experiment ID: `7242f5f7`*
- **Global Speed Winner**: `cfg_Q4_K_M_T6_C4096`
  - Mean Latency: **`1801.65 ms`** (**32.56% improvement** vs baseline `2671.42 ms`)
  - Mean Throughput: **`48.14 tok/s`** (**47.08% improvement** vs baseline `32.73 tok/s`)
  - Model Footprint: **`468.6 MB`**
- **Pareto Frontier**: `cfg_Q4_K_M_T6_C4096` strictly dominates all other 35 configurations for short generation.

### B. Context Stress Workload (`context_stress`, 36 Configs)
*Experiment ID: `1e6bc7d2`*
- **Global Speed & Throughput Winner**: `cfg_Q8_0_T8_C2048`
  - Mean Latency: **`4310.92 ms`** (**20.35% improvement** vs baseline `5412.55 ms`)
  - Mean Throughput: **`25.82 tok/s`** (**28.72% improvement** vs baseline `20.06 tok/s`)
  - Model Footprint: **`644.4 MB`**
- **Global Size Winner**: `cfg_Q4_K_M_T4_C1024`
  - Mean Latency: **`4381.38 ms`**
  - Mean Throughput: **`24.32 tok/s`**
  - Model Footprint: **`468.6 MB`** (27.3% smaller footprint than Q8_0)
- **Pareto Frontier**: `cfg_Q8_0_T8_C2048` and `cfg_Q4_K_M_T4_C1024`.

---

## 6. Platform Attribution & Scientific Safety
- **Host Platform**: `DEVELOPMENT — x86_64` (AMD64 6 physical / 12 logical cores, 15.42 GB RAM).
- **Disclaimer**: These measurements are **NOT Google Axion / Arm64 measurements**.
- **Quality Score Policy**: `quality_score: null` is explicitly recorded because model accuracy was unmeasured during latency sweeps.

---

## 7. Reproducibility
```powershell
# 1. Run Short-Generation Global Sweep (36 Configurations)
python tools/optimize.py --dimension global --workload short_generation --objective speed

# 2. Run Context-Stress Global Sweep (36 Configurations)
python tools/optimize.py --dimension global --workload context_stress --objective speed

# 3. Launch Web Dashboard
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```
