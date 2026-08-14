import os
import json
import csv
import datetime
import statistics
from typing import List, Dict, Any, Optional
from enum import Enum

from backend.inference.llama_cpp_engine import LlamaCppEngine
from backend.platform.detector import get_platform
from backend.config import config

class WorkloadType(str, Enum):
    SHORT_GENERATION = "short_generation"
    CONTEXT_STRESS = "context_stress"

# Workload A: Standard Short Generation (5 prompts)
SHORT_GENERATION_PROMPTS = [
    "What is the capital of France?",
    "Explain the theory of relativity in one simple paragraph.",
    "Write a Python function to calculate the Fibonacci sequence.",
    "Summarize the plot of Romeo and Juliet in 3 sentences.",
    "List 5 benefits of using Arm64 processors in cloud computing."
]

# Workload B: Context-Stress Workload (~650 tokens context)
CONTEXT_STRESS_DOCUMENT = """
Cloud computing architectures are undergoing a generational shift toward custom Arm-based silicon. 
Traditional x86 architectures have dominated data centers for decades with complex out-of-order execution 
pipelines and simultaneous multithreading (SMT). However, thermal limits, core scaling barriers, and energy 
density constraints have led cloud hyperscalers to deploy custom processors built on the Arm Neoverse platform.

Google Axion represents Google Cloud's first dedicated Arm-based CPU family, designed on the Arm Neoverse V2 
architecture. Axion processors provide high single-threaded performance, full vCPU-to-physical-core isolation, 
and hardware-level vector extensions (SVE2 and Bfloat16). Unlike SMT systems where two threads contend for the 
same execution pipeline and L1/L2 caches, dedicated physical cores ensure deterministic latency for demanding 
AI inference and microservice workloads.

When optimizing large language model inference on CPU architectures, several operational variables dictate 
efficiency: thread allocation, context window allocation (KV-cache footprint), memory bandwidth saturation, 
and numerical quantization. Memory bandwidth is typically the primary bottleneck in autoregressive token 
generation. Thread counts that exceed physical core bounds introduce cache thrashing and context switching 
overheads, while excessively large context allocations consume critical cache lines and memory bandwidth. 

Automated inference optimizers must systematically profile these trade-offs to map the Pareto frontier across 
latency, throughput, and memory consumption.
"""

CONTEXT_STRESS_PROMPTS = [
    f"Based on the following document, explain why hyperscalers are adopting Arm-based CPUs:\n\n{CONTEXT_STRESS_DOCUMENT}\n\nSummary:",
    f"Based on the following document, what is the architectural difference between Google Axion cores and traditional SMT threads?\n\n{CONTEXT_STRESS_DOCUMENT}\n\nAnswer:",
    f"Based on the following document, identify the primary bottleneck in autoregressive LLM CPU inference and explain why:\n\n{CONTEXT_STRESS_DOCUMENT}\n\nExplanation:",
    f"Based on the following document, what happens when thread counts exceed physical core bounds?\n\n{CONTEXT_STRESS_DOCUMENT}\n\nAnalysis:",
    f"Based on the following document, summarize the three key operational variables that dictate CPU inference efficiency in 3 bullet points:\n\n{CONTEXT_STRESS_DOCUMENT}\n\nBullet points:"
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
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.results_dir = os.path.join(self.root_dir, "benchmarks", "results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def run_baseline(
        self, 
        threads: int = None, 
        context_size: int = None,
        quantization: str = None,
        workload_type: WorkloadType = WorkloadType.SHORT_GENERATION,
        save: bool = True
    ) -> Dict[str, Any]:
        """Runs a benchmark sweep with specified workload and configuration."""
        if not self.engine.is_loaded:
            raise RuntimeError("Cannot benchmark: Model not loaded")

        prompts = SHORT_GENERATION_PROMPTS if workload_type == WorkloadType.SHORT_GENERATION else CONTEXT_STRESS_PROMPTS
        
        used_threads = threads if threads is not None else getattr(self.engine, "active_threads", config.MODEL_THREADS)
        used_context = context_size if context_size is not None else getattr(self.engine, "active_context_size", config.MODEL_CONTEXT_SIZE)
        used_quant = quantization if quantization is not None else getattr(self.engine, "active_quantization", "Q4_K_M")
        
        print(f"Starting benchmark run [{workload_type.value.upper()}] (Quant: {used_quant}, Threads: {used_threads}, Context: {used_context})...")
        
        # 1. Warmup runs (2 runs)
        print("  Running warmups (2 runs)...")
        for i in range(2):
            self.engine.generate(prompts[i], max_tokens=10, temperature=0.0)
            
        # 2. Measured runs (5 runs)
        latencies = []
        tokens_per_sec_list = []
        all_results = []
        
        print("  Running measured benchmarks (5 runs)...")
        for idx, prompt in enumerate(prompts):
            print(f"    Run {idx+1}/{len(prompts)}...")
            res = self.engine.generate(prompt, max_tokens=config.MAX_TOKENS, temperature=0.0)
            
            latencies.append(res['latency_ms'])
            tokens_per_sec_list.append(res['tokens_per_second'])
            all_results.append({
                "run_index": idx + 1,
                "prompt": prompt[:80] + ("..." if len(prompt) > 80 else ""),
                "latency_ms": res['latency_ms'],
                "tokens_generated": res['tokens_generated'],
                "tokens_per_second": res['tokens_per_second']
            })
            
        # 3. Statistical Calculations
        platform_info = get_platform().to_dict()
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
        
        benchmark_report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "platform": platform_info,
            "architecture": platform_info["architecture"],
            "cpu": platform_info["cpu"],
            "cores": platform_info["physical_cores"],
            "model": model_info.get("repository", "Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
            "model_filename": model_info.get("filename", "model.gguf"),
            "model_filepath": model_info.get("filepath"),
            "model_size_mb": model_info.get("model_size_mb", 0.0),
            "file_size_bytes": model_info.get("file_size_bytes"),
            "sha256": model_info.get("sha256"),
            "license": model_info.get("license", "Apache-2.0"),
            "quality_score": model_info.get("quality_score"),
            "load_time_ms": model_info.get("load_time_ms"),
            "runtime": model_info.get("runtime", "llama.cpp"),
            "workload": {
                "type": workload_type.value,
                "prompt_suite_version": "v1.0",
                "prompts_count": len(prompts),
                "estimated_input_tokens_per_prompt": 15 if workload_type == WorkloadType.SHORT_GENERATION else 650,
                "max_tokens": config.MAX_TOKENS,
                "temperature": 0.0,
                "token_accounting": "completion_tokens_only"
            },
            "configuration": {
                "quantization": model_info.get("quantization", used_quant),
                "threads": used_threads,
                "context_size": used_context,
                "batch_size": 1,
                "model_size_mb": model_info.get("model_size_mb", 0.0)
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
                "min_latency_ms": min_latency,
                "max_latency_ms": max_latency,
                "std_latency_ms": std_latency,
                "mean_tokens_per_second": mean_tps,
                "median_tokens_per_second": median_tps,
                "p95_tokens_per_second": p95_tps,
                "min_tokens_per_second": min_tps,
                "max_tokens_per_second": max_tps,
                "std_tokens_per_second": std_tps,
                "model_size_mb": model_info["model_size_mb"],
                "load_time_ms": model_info.get("load_time_ms"),
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
        w_type = report["workload"]["type"]
        q_type = report["configuration"]["quantization"]
        
        # JSON
        json_path = os.path.join(self.results_dir, f"benchmark_{w_type}_{q_type}_{timestamp_str}.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # CSV
        csv_path = os.path.join(self.results_dir, f"benchmark_{w_type}_{q_type}_{timestamp_str}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Workload", "Provider", "Architecture", "CPU", "Cores", "Model", "Quantization", "Threads", "Context",
                "Model_Size_MB", "Load_Time_ms", "SHA256",
                "Mean_Latency_ms", "Median_Latency_ms", "P95_Latency_ms", "Min_Latency_ms", "Max_Latency_ms", "Std_Latency_ms",
                "Mean_TPS", "Median_TPS", "P95_TPS", "Min_TPS", "Max_TPS", "Std_TPS",
                "Memory_MB", "Memory_Status"
            ])
            writer.writerow([
                report["timestamp"],
                report["workload"]["type"],
                report["platform"]["provider"],
                report["platform"]["architecture"],
                report["platform"]["cpu"],
                report["platform"]["physical_cores"],
                report["model"],
                report["configuration"]["quantization"],
                report["configuration"]["threads"],
                report["configuration"]["context_size"],
                report["model_size_mb"],
                report.get("load_time_ms"),
                report.get("sha256"),
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
            
        print(f"  Results saved to {json_path} and {csv_path}")
