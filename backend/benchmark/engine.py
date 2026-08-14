import os
import json
import csv
import datetime
import statistics
from typing import List, Dict, Any
from backend.inference.llama_cpp_engine import LlamaCppEngine
from backend.utils.system import get_system_info
from backend.config import config

PROMPTS = [
    "What is the capital of France?",
    "Explain the theory of relativity in one simple paragraph.",
    "Write a Python function to calculate the Fibonacci sequence.",
    "Summarize the plot of Romeo and Juliet in 3 sentences.",
    "List 5 benefits of using Arm64 processors in cloud computing."
]

class BenchmarkEngine:
    def __init__(self, engine: LlamaCppEngine):
        self.engine = engine
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmarks", "results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def run_baseline(self, threads: int = None) -> Dict[str, Any]:
        """Runs the baseline benchmark."""
        if not self.engine.is_loaded:
            raise RuntimeError("Cannot benchmark: Model not loaded")

        if threads is not None:
            # Reconfigure engine if needed, but for now we assume it's loaded with correct threads or we just pass it to generator if possible.
            # llama.cpp requires setting threads on init. We will assume the engine was initialized with the desired thread count via config.
            pass
            
        print("Starting baseline benchmark...")
        
        # 1. Warmup runs
        print("Running warmup (2 runs)...")
        for i in range(2):
            self.engine.generate(PROMPTS[i], max_tokens=10, temperature=0.0)
            
        # 2. Measured runs
        latencies = []
        tokens_per_sec_list = []
        all_results = []
        
        print("Running measured benchmarks (5 runs)...")
        for idx, prompt in enumerate(PROMPTS):
            print(f"  Run {idx+1}/{len(PROMPTS)}...")
            res = self.engine.generate(prompt, max_tokens=config.MAX_TOKENS, temperature=0.0)
            
            latencies.append(res['latency_ms'])
            tokens_per_sec_list.append(res['tokens_per_second'])
            all_results.append({
                "prompt": prompt,
                "latency_ms": res['latency_ms'],
                "tokens_generated": res['tokens_generated'],
                "tokens_per_second": res['tokens_per_second']
            })
            
        # 3. Calculate metrics
        sys_info = get_system_info()
        model_info = self.engine.get_model_info()
        
        latencies.sort()
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        
        # P95 calculation
        p95_idx = int(0.95 * len(latencies))
        if p95_idx >= len(latencies): p95_idx = len(latencies) - 1
        p95_latency = latencies[p95_idx]
        
        mean_tps = statistics.mean(tokens_per_sec_list)
        
        benchmark_report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "architecture": sys_info["architecture"],
            "cpu": sys_info["cpu"],
            "cores": sys_info["cpu_cores_physical"],
            "ram_gb": sys_info["ram_gb"],
            "model": model_info["repository"],
            "model_size_mb": model_info["model_size_mb"],
            "runtime": model_info["runtime"],
            "configuration": {
                "quantization": model_info["quantization"],
                "threads": config.MODEL_THREADS,
                "batch_size": 1,
                "context_size": config.MODEL_CONTEXT_SIZE
            },
            "benchmark_parameters": {
                "warmups": 2,
                "measured_runs": 5,
                "max_tokens": config.MAX_TOKENS
            },
            "results": {
                "mean_latency_ms": mean_latency,
                "median_latency_ms": median_latency,
                "p95_latency_ms": p95_latency,
                "mean_tokens_per_second": mean_tps,
                "runs": all_results
            }
        }
        
        self.save_results(benchmark_report)
        return benchmark_report

    def save_results(self, report: Dict[str, Any]):
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON
        json_path = os.path.join(self.results_dir, f"benchmark_{timestamp_str}.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # CSV
        csv_path = os.path.join(self.results_dir, f"benchmark_{timestamp_str}.csv")
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Architecture", "CPU", "Cores", "Model", "Quantization", "Threads", "Mean_Latency_ms", "Median_Latency_ms", "P95_Latency_ms", "Mean_TPS"])
            writer.writerow([
                report["timestamp"],
                report["architecture"],
                report["cpu"],
                report["cores"],
                report["model"],
                report["configuration"]["quantization"],
                report["configuration"]["threads"],
                report["results"]["mean_latency_ms"],
                report["results"]["median_latency_ms"],
                report["results"]["p95_latency_ms"],
                report["results"]["mean_tokens_per_second"]
            ])
            
        print(f"Results saved to {json_path} and {csv_path}")
