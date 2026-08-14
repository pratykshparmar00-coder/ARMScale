# ARMScale
**Autonomous Arm64 AI Inference Optimizer**

ARMScale is an autonomous optimization and benchmarking platform designed to extract maximum price-performance efficiency from Arm64 cloud infrastructure (such as Google Axion C4A, AWS Graviton, and Ampere Altra) for AI inference workloads.

---

## Key Features
- **Cloud-Agnostic Platform Abstraction**: Seamlessly operates across local workstations, Google Cloud Axion instances, and generic Arm64 bare metal.
- **Joint Multi-Dimensional Optimization**: Evaluates joint trade-offs between CPU threads (`2, 4, 6, 8`) and context window sizes (`1024, 2048, 4096`).
- **Workload Separation**: Distinct evaluation for `short_generation` interactive latency vs. `context_stress` long-context prefill workloads.
- **Objective-Driven Recommendations**: Recommends optimal configurations based on user priorities (Speed, Throughput, or Balanced).
- **Empirical Pareto Analysis**: Mathematically computes non-dominated frontier configurations.
- **Live Minimal Web Dashboard**: Interactive visualization of the Latency vs. Throughput trade-off space with real measurement points.
- **Zero-Fabrication Architecture**: Strict adherence to real monotonic timing and raw measurement preservation.

---

## Quickstart

### 1. Setup Environment
```bash
python -m venv .venv
# On Windows: .venv\Scripts\Activate.ps1
# On Linux: source .venv/bin/activate
pip install -r requirements.txt
python tools/download_model.py
```

### 2. Run Optimization Experiments
```bash
# Run 12-configuration joint sweep on short generation
python tools/optimize.py --dimension combined --workload short_generation --objective speed

# Run 12-configuration joint sweep on context stress
python tools/optimize.py --dimension combined --workload context_stress --objective speed
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
    API --> OPT[Optimization Engine]
    OPT --> GEN[Configuration Generator]
    OPT --> BENCH[Benchmark Engine]
    BENCH --> ENG[Inference Engine (llama.cpp)]
    OPT --> SCORE[Scoring & Pareto Engine]
    BENCH --> PLAT[Platform Adapter Layer]
```

---

## Documentation
- [Architecture](docs/ARCHITECTURE.md)
- [Platform Abstraction](docs/PLATFORM_ARCHITECTURE.md)
- [Multi-Dimensional Optimization](docs/MULTI_DIMENSIONAL_OPTIMIZATION.md)
- [Context Optimization](docs/CONTEXT_OPTIMIZATION.md)
- [Google Axion Target](docs/GOOGLE_AXION.md)
- [Axion Deployment Guide](docs/DEPLOYMENT_AXION.md)

---

## License
MIT License
