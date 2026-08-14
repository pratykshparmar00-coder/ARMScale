# ARMScale Quantization Optimization & Multi-Variant Benchmarking

## Overview
Numerical quantization reduces parameter bit-width, reducing model weight memory footprint and memory bandwidth pressure during autoregressive decode steps. However, different quantization methods (e.g. `Q4_K_M`, `Q5_K_M`, `Q8_0`) exhibit distinct tradeoffs between file size, load time, dequantization compute overhead, and inference throughput.

ARMScale implements **Quantization** as a primary optimization dimension, evaluating exact variants from the same model family under controlled experimental conditions.

---

## Verified Model Variants Catalog
All variants belong strictly to the `Qwen/Qwen2.5-0.5B-Instruct-GGUF` model repository and are verified via cryptographic SHA256 checksums:

| Variant | Exact Filename | Exact File Size | SHA256 Checksum | License |
| :--- | :--- | :--- | :--- | :--- |
| **Q4_K_M** | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 491,400,032 bytes (468.64 MB) | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` | Apache-2.0 |
| **Q5_K_M** | `qwen2.5-0.5b-instruct-q5_k_m.gguf` | 522,186,592 bytes (498.00 MB) | `041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55` | Apache-2.0 |
| **Q8_0** | `qwen2.5-0.5b-instruct-q8_0.gguf` | 675,710,816 bytes (644.41 MB) | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` | Apache-2.0 |

---

## Multi-Objective Pareto Dominance
When evaluating across quantization formats, ARMScale calculates Pareto non-dominance across three measurable dimensions:
1. **Mean Latency ($ms$)**: Lower is better ($\downarrow$)
2. **Mean Throughput ($tok/s$)**: Higher is better ($\uparrow$)
3. **Model Size ($MB$)**: Lower is better ($\downarrow$)

A configuration $A$ strictly dominates $B$ ($A \succ B$) if:
$$(\text{Latency}_A \le \text{Latency}_B) \land (\text{Throughput}_A \ge \text{Throughput}_B) \land (\text{Size}_A \le \text{Size}_B)$$
$$\text{with at least one strict inequality.}$$

---

## Scientific Rigor & Quality Policy
- **Quality Score Unmeasured**: ARMScale explicitly marks `quality_score: null` until standard evaluation benchmarks (e.g. MMLU, GSM8K, perplexity) are measured. We never assume or claim quality differences without empirical testing.
- **Identical Conditions**: Every quantization variant is evaluated using the identical prompt sequence, token limits, temperature (`0.0`), thread allocation (`6`), and context size (`4096`).
