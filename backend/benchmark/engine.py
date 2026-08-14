import os
import json
import csv
import datetime
import statistics
from typing import List, Dict, Any, Optional
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

def calculate_percentile(data: List[float], p: float) -> float:
    """Calculates the p-th percentile of a list of floats using linear interpolation."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    d = k - f
    return s[f] + (s[c] - s[f]) * d

class BenchmarkEngine:
    def __init__(self, engine: LlamaCppEngine):
        self.engine = engine
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmarks", "results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def run_baseline(self, threads: int = None, save: bool = True) -> Dict[str, Any]:
        """Runs the baseline benchmark with rigorous statistical metrics."""
        if not self.engine.is_loaded:
            raise RuntimeError("Cannot benchmark: Model not loaded")

        print("Starting benchmark run...")
        
        # 1. Warmup runs (2 runs)
        print("Running warmup (2 runs)...")
        for i in range(2):
            self.engine.generate(PROMPTS[i], max_tokens=10, temperature=0.0)
            
        # 2. Measured runs (5 runs)
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
                "run_index": idx + 1,
                "prompt": prompt,
                "latency_ms": res['latency_ms'],
                "tokens_generated": res['tokens_generated'],
                "tokens_per_second": res['tokens_per_second']
            })
            
        # 3. Calculate metrics directly from raw runs
        sys_info = get_system_info()
        model_info = self.engine.get_model_info()
        
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = calculate_percentile(latencies, 95.0)
        min_latency = min(latencies)
        max_latency = max(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        
        mean_tps = statistics.mean(tokens_per_sec_list)
        median_tps = statistics.median(tokens_per_sec_list)
        p95_tps = calculate_percentile(tokens_per_sec_list, 95.0)
        min_tps = min(tokens_per_sec_list)
        max_tps = max(tokens_per_sec_list)
        std_tps = statistics.stdev(tokens_per_sec_list) if len(tokens_per_sec_list) > 1 else 0.0
        
        used_threads = threads if threads is not None else config.MODEL_THREADS
        
        benchmark_report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "architecture": sys_info["architecture"],
            "cpu": sys_info["cpu"],
            "cores": sys_info["cpu_cores_physical"],
            "logical_cores": sys_info.get("cpu_cores_logical", sys_info["cpu_cores_physical"]),
            "ram_gb": sys_info["ram_gb"],
            "model": model_info["repository"],
            "model_filename": model_info["filename"],
            "model_size_mb": model_info["model_size_mb"],
            "runtime": model_info["runtime"],
            "configuration": {
                "quantization": model_info["quantization"],
                "threads": used_threads,
                "batch_size": 1,
                "context_size": config.MODEL_CONTEXT_SIZE
            },
            "benchmark_parameters": {
                "warmups": 2,
                "measured_runs": 5,
                "max_tokens": config.MAX_TOKENS,
                "temperature": 0.0,
                "token_accounting": "completion_tokens_only"
            },
            "results": {
                "mean_latency_ms": mean_latency,
                "median_latency_ms": median_latency,
                "p95_latency_ms": p95_latency,
                "min_latency_ms": min_latency,
                "max_latency_ms": max_latency,
                "std_latency_ms": std_latency,
                "mean_tokens_per_second": mean_tps,
                "median_tokens_per_second": median_tps,
                "p95_tokens_per_second": p95_tps,
                "min_tokens_per_second": min_tps,
                "max_tokens_per_second": max_tps,
                "std_tokens_per_second": std_tps,
                "memory_mb": None,
                "memory_status": "unavailable",
                "runs": all_results
            }
        }
        
        if save:
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
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Architecture", "CPU", "Cores", "Model", "Quantization", "Threads",
                "Mean_Latency_ms", "Median_Latency_ms", "P95_Latency_ms", "Min_Latency_ms", "Max_Latency_ms", "Std_Latency_ms",
                "Mean_TPS", "Median_TPS", "P95_TPS", "Min_TPS", "Max_TPS", "Std_TPS",
                "Memory_MB", "Memory_Status"
            ])
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
                report["results"]["min_latency_ms"],
                report["results"]["max_latency_ms"],
                report["results"]["std_latency_ms"],
                report["results"]["mean_tokens_per_second"],
                report["results"]["median_tokens_per_second"],
                report["results"]["p95_tokens_per_second"],
                report["results"]["min_tokens_per_second"],
                report["results"]["max_tokens_per_second"],
                report["results"]["std_tokens_per_second"],
                report["results"]["memory_mb"],
                report["results"]["memory_status"]
            ])
            
        print(f"Results saved to {json_path} and {csv_path}")
