# Migration to Cloud AI

## Pivot from Mobile AI

The previous direction of this project was focused on Mobile AI and local AI architectures, including iOS/Android mobile assistants and LLaVA integrations. 

That direction has been discontinued.

The repository was intentionally restarted clean. The old codebase was entirely removed. 

## ARMScale: Cloud AI Optimization

ARMScale now focuses strictly on **Cloud AI**.

The target domain is **Arm64 cloud inference optimization**.

Our architecture is designed to:
1. Deploy an AI model on Arm64 cloud infrastructure (e.g., AWS Graviton, Google Axion, Microsoft Cobalt).
2. Profile inference performance (latency, throughput, memory).
3. Test multiple valid configurations automatically.
4. Select the best configuration according to user-defined objectives (SPEED, THROUGHPUT, MEMORY, BALANCED, COST).
5. Demonstrate reproducible baseline vs optimized results.

Since the old code was removed, there is no direct migration of code. The migration represents a conceptual pivot to cloud architecture and performance benchmarking on Arm64.
