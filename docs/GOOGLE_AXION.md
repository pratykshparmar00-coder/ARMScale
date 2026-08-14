# Google Axion Arm64 Cloud Target

## Overview
Google Axion is Google Cloud's custom Arm-based CPU processor, built on the Arm Neoverse V2 architecture. It powers the **C4A** machine series in Google Compute Engine, delivering high compute performance and energy efficiency for general-purpose workloads, cloud infrastructure, and AI inference.

## Architecture Specifications
- **Processor**: Google Axion (custom Arm-based Neoverse V2 core architecture)
- **Machine Family**: `c4a-standard-*` (e.g., `c4a-standard-4`, `c4a-standard-8`, `c4a-standard-16`)
- **ISA**: ARMv9-A with SVE2 and Bfloat16 support
- **Architecture String**: `aarch64` / `arm64`
- **Supported OS**: Ubuntu 22.04 / 24.04 LTS (Arm64), Debian 12 (Arm64)
- **Hyperthreading**: 1 vCPU per physical core (no SMT overhead, providing predictable multi-threaded inference)

## Benchmark Configuration
- **Model**: `Qwen2.5-0.5B-Instruct-GGUF` (`qwen2.5-0.5b-instruct-q4_k_m.gguf`)
- **Quantization**: `Q4_K_M`
- **Runtime**: `llama-cpp-python` compiled natively for `aarch64` with NEON / SVE acceleration
- **Prompt Suite**: Standardized 5-prompt suite (identical to x86_64 baseline)
- **Benchmark Parameters**: 2 warmups, 5 measured runs, `max_tokens = 128`, `temperature = 0.0`

## Environmental Verification
When running on Google Compute Engine, ARMScale automatically queries the GCP Metadata server (`http://metadata.google.internal/computeMetadata/v1/`) to dynamically detect:
- `cloud_provider`: Google Cloud
- `machine_family`: C4A
- `processor_family`: Google Axion
- `machine_type`: (e.g., `c4a-standard-4`)
- `zone`: (e.g., `us-central1-a`)
