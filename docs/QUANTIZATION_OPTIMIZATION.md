# ARMScale Quantization Optimization & Multi-Variant Benchmarking

## 1. Why Quantization Matters in Cloud AI Inference
Autoregressive Large Language Model (LLM) inference on CPU architectures is memory-bandwidth bound. During generation, every decoded token requires transferring the entire model parameter weight matrix from system RAM to CPU registers and cache hierarchies.

Numerical quantization compresses model weights from 16-bit floating point down to lower bit-depth representations:
- **Reduced Memory Bandwidth Consumption**: Transferring 4 or 8 bits per weight drastically decreases the gigabytes/second demanded from the memory bus.
- **Cache Locality**: Smaller footprint allows greater portions of model weights to fit within L3 cache partitions.
- **De-quantization Trade-offs**: In exchange for reduced memory traffic, CPU instruction pipelines must perform lightweight on-the-fly dequantization arithmetic (e.g. AVX2 / ARM NEON / SVE2 vector instructions).

---

## 2. Verified Model Variants Catalog
All variants belong strictly to the official `Qwen/Qwen2.5-0.5B-Instruct-GGUF` model repository and are cryptographically verified via SHA256 hashes:

| Variant | Exact Filename | Exact File Size | SHA256 Checksum | License |
| :--- | :--- | :--- | :--- | :--- |
| **Q4_K_M** | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 491,400,032 bytes (468.64 MB) | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` | Apache-2.0 |
| **Q5_K_M** | `qwen2.5-0.5b-instruct-q5_k_m.gguf` | 522,186,592 bytes (498.00 MB) | `041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55` | Apache-2.0 |
| **Q8_0** | `qwen2.5-0.5b-instruct-q8_0.gguf` | 675,710,816 bytes (644.41 MB) | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` | Apache-2.0 |

---

## 3. Measured Trade-Offs (x86_64 Development Benchmark)

### Short-Generation Workload (`short_generation`, T=6, Ctx=4096)
- **`Q8_0`** ($644.4\text{ MB}$, $598\text{ ms load}$): **$2237.84\text{ ms}$ mean latency**, **$38.84\text{ tok/s}$**.
- **`Q5_K_M`** ($498.0\text{ MB}$, $619\text{ ms load}$): **$2402.29\text{ ms}$ mean latency**, **$35.61\text{ tok/s}$**.
- **`Q4_K_M`** ($468.6\text{ MB}$, $868\text{ ms load}$): **$2916.56\text{ ms}$ mean latency**, **$35.60\text{ tok/s}$**.

### Context-Stress Workload (`context_stress`, ~650 input tokens, T=6, Ctx=4096)
- **`Q8_0`** ($644.4\text{ MB}$, $677\text{ ms load}$): **$5364.93\text{ ms}$ mean latency**, **$21.10\text{ tok/s}$**.
- **`Q4_K_M`** ($468.6\text{ MB}$, $546\text{ ms load}$): **$6291.11\text{ ms}$ mean latency**, **$16.96\text{ tok/s}$**.
- **`Q5_K_M`** ($498.0\text{ MB}$, $474\text{ ms load}$): **$7470.30\text{ ms}$ mean latency**, **$15.80\text{ tok/s}$**.

---

## 4. Multi-Objective 3D Pareto Frontier
When evaluating across quantization formats, ARMScale evaluates Pareto non-dominance across three measurable dimensions:
1. **Mean Latency ($ms$)**: Lower is better ($\downarrow$)
2. **Mean Throughput ($tok/s$)**: Higher is better ($\uparrow$)
3. **Model Size ($MB$)**: Lower is better ($\downarrow$)

A candidate configuration $A$ strictly dominates $B$ ($A \succ B$) if and only if:
$$(\text{Latency}_A \le \text{Latency}_B) \land (\text{Throughput}_A \ge \text{Throughput}_B) \land (\text{Size}_A \le \text{Size}_B)$$
$$\text{with at least one strict inequality: } (\text{Latency}_A < \text{Latency}_B) \lor (\text{Throughput}_A > \text{Throughput}_B) \lor (\text{Size}_A < \text{Size}_B)$$

Configurations where no other configuration strictly dominates are flagged `pareto_optimal: true`.

---

## 5. Scientific Rigor & Quality Policy
- **Zero Quality Claims Without Empirical Testing**: ARMScale explicitly marks `quality_score: null` until standard evaluation benchmarks (e.g. MMLU, GSM8K, perplexity) are measured. We never assume or claim quality differences without empirical testing.
- **Strictly Controlled Conditions**: Every quantization variant is evaluated using the identical prompt sequence, token limits, temperature (`0.0`), thread allocation (`6`), and context size (`4096`).
- **Separation of Architectures**: All experiments on development machines are strictly tagged `DEVELOPMENT — x86_64` and never represented as Arm64 / Google Axion measurements.

---

## 6. Reproducibility Instructions
To reproduce the exact experiments:

```bash
# 1. Activate environment
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1

# 2. Download and verify model variants
python tools/download_model.py --variant Q5_K_M
python tools/download_model.py --variant Q8_0

# 3. Run Short-Generation Quantization Sweep
python tools/optimize.py --dimension quantization --workload short_generation --threads 6 --contexts 4096 --quantizations Q4_K_M,Q5_K_M,Q8_0 --objective speed

# 4. Run Context-Stress Quantization Sweep
python tools/optimize.py --dimension quantization --workload context_stress --threads 6 --contexts 4096 --quantizations Q4_K_M,Q5_K_M,Q8_0 --objective speed
```
