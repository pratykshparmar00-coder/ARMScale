# ARMScale
**Autonomous Arm64 AI Inference Optimizer**

ARMScale is an autonomous optimization and benchmarking platform designed to extract maximum price-performance efficiency from Arm64 cloud infrastructure (such as Google Axion C4A, AWS Graviton, and Ampere Altra) for AI inference workloads.

---

## Key Features
- **Cloud-Agnostic Platform Abstraction**: Seamlessly operates across local workstations, Google Cloud Axion instances, and generic Arm64 bare metal.
- **Autonomous Global Multi-Dimensional Optimization**: Evaluates the full 36-configuration Cartesian product across Quantizations (`Q4_K_M`, `Q5_K_M`, `Q8_0`) $\times$ CPU Threads (`2, 4, 6, 8`) $\times$ Context Windows (`1024, 2048, 4096`).
- **Workload Separation**: Distinct evaluation for `short_generation` interactive latency vs. `context_stress` long-context prefill workloads.
- **Objective-Driven Recommendations**: Recommends optimal configurations based on user priorities (`Speed`, `Throughput`, `Size`, `Balanced`).
- **Empirical 3D Pareto Analysis**: Mathematically computes non-dominated frontier configurations across Latency $\downarrow$, Throughput $\uparrow$, and Model Size $\downarrow$.
- **Live Minimal Web Dashboard**: Interactive visualization of the Latency vs. Throughput trade-off space with real measurement points and model footprint scaling.
- **Zero-Fabrication Architecture**: Strict adherence to real monotonic timing and raw measurement preservation.

---

## Quickstart

### 1. Setup Environment
```bash
python -m venv .venv
# On Windows: .venv\Scripts\Activate.ps1
# On Linux: source .venv/bin/activate
pip install -r requirements.txt
python tools/download_model.py --variant all
```

### 2. Run Global Optimization Sweeps (36 Configurations each)
```bash
# Run 36-configuration global sweep on short generation
python tools/optimize.py --dimension global --workload short_generation --objective speed

# Run 36-configuration global sweep on context stress
python tools/optimize.py --dimension global --workload context_stress --objective speed
```

### 3. Launch the Web UI & API
```bash
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```
Open `http://localhost:8000/` in your browser to explore the dashboard.

---

## Architecture

```mermaid
graph TD
    API[FastAPI Gateway / Web UI] --> REC[Recommendation Engine]
    API --> REG[Experiment Registry]
    API --> OPT[Optimization Engine]
    OPT --> GEN[Configuration Generator]
    OPT --> BENCH[Benchmark Engine]
    BENCH --> ENG[Inference Engine (llama.cpp)]
    OPT --> SCORE[Scoring & 3D Pareto Engine]
    BENCH --> PLAT[Platform Adapter Layer]
```

---

## Documentation
- [Global Optimization](docs/GLOBAL_OPTIMIZATION.md)
- [Quantization Optimization](docs/QUANTIZATION_OPTIMIZATION.md)
- [Multi-Dimensional Optimization](docs/MULTI_DIMENSIONAL_OPTIMIZATION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Platform Abstraction](docs/PLATFORM_ARCHITECTURE.md)
- [Google Axion Target](docs/GOOGLE_AXION.md)
- [Axion Deployment Guide](docs/DEPLOYMENT_AXION.md)

---

## License
MIT License
