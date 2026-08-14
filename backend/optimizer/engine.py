import os
import time
import uuid
import json
import glob
import datetime
from typing import Dict, Any, List, Optional

from .models import OptimizationRequest, OptimizationResult, OptimizationConfig, Objective
from .config_generator import ConfigurationGenerator
from .scoring import ScoringEngine
from backend.benchmark.engine import BenchmarkEngine
from backend.inference.llama_cpp_engine import LlamaCppEngine
from backend.config import config

class OptimizationEngine:
    def __init__(self, inference_engine: LlamaCppEngine, benchmark_engine: BenchmarkEngine):
        self.inference_engine = inference_engine
        self.benchmark_engine = benchmark_engine
        self.config_generator = ConfigurationGenerator()
        self.scoring_engine = ScoringEngine()
        
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.results_dir = os.path.join(self.root_dir, "benchmarks", "results", "optimization")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        self.jobs = {}

    def get_reference_baseline(self) -> Optional[Dict[str, Any]]:
        """Finds the canonical baseline benchmark recorded (filtering out mock test runs)."""
        baseline_dir = os.path.join(self.root_dir, "benchmarks", "results")
        json_files = sorted(glob.glob(os.path.join(baseline_dir, "benchmark_*.json")))
        for fpath in json_files:
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    if data.get("model") != "mock/mock" and data.get("runtime") != "mock":
                        return data
            except Exception as e:
                print(f"Warning: Could not read baseline candidate {fpath}: {e}")
        return None

    def start_optimization(self, request: OptimizationRequest) -> str:
        experiment_id = str(uuid.uuid4())[:8]
        self.jobs[experiment_id] = {
            "status": "queued",
            "completed": 0,
            "total": 0,
            "current_configuration": None,
            "result": None
        }
        return experiment_id

    def run_optimization_sync(self, request: OptimizationRequest, experiment_id: str) -> OptimizationResult:
        start_time = time.time()
        self.jobs[experiment_id]["status"] = "running"
        
        # 1. Retrieve Canonical Reference Baseline (Phase C)
        ref_baseline = self.get_reference_baseline()
        original_threads = config.MODEL_THREADS
        
        if not ref_baseline:
            print("No existing baseline found. Running baseline benchmark now...")
            ref_baseline = self.benchmark_engine.run_baseline(threads=original_threads, save=True)
        else:
            print(f"Referencing recorded baseline (Mean Latency: {ref_baseline['results']['mean_latency_ms']:.2f}ms, Mean Throughput: {ref_baseline['results']['mean_tokens_per_second']:.2f} tok/s)")
            
        # 2. Generate Candidate Configurations
        configs = self.config_generator.generate_configurations(request.threads_to_test)
        self.jobs[experiment_id]["total"] = len(configs)
        
        results = []
        
        # 3. Benchmark Each Candidate Configuration
        for idx, cfg in enumerate(configs):
            self.jobs[experiment_id]["completed"] = idx
            self.jobs[experiment_id]["current_configuration"] = cfg.dict()
            
            print(f"\n[{idx+1}/{len(configs)}] Testing thread configuration: {cfg.threads} threads")
            
            # Safely unload and reload model with new thread count
            self.inference_engine.unload_model()
            config.MODEL_THREADS = cfg.threads
            
            success = self.inference_engine.load_model()
            if not success:
                print(f"Failed to load model for {cfg.threads} threads, skipping.")
                continue
                
            # Run benchmark (saves individual benchmark artifact)
            bench_res = self.benchmark_engine.run_baseline(threads=cfg.threads, save=True)
            bench_res['configuration']['threads'] = cfg.threads
            results.append(bench_res)
            
        # 4. Score and Analyze Results
        best_cfg = self.scoring_engine.score_results(results, request.objective)
        pareto_cfgs = self.scoring_engine.get_pareto_frontier(results)
        
        # 5. Calculate Improvements vs Reference Baseline
        baseline_lat = ref_baseline['results']['mean_latency_ms']
        baseline_tps = ref_baseline['results']['mean_tokens_per_second']
        
        best_lat = best_cfg['results']['mean_latency_ms']
        best_tps = best_cfg['results']['mean_tokens_per_second']
        
        # Latency improvement: lower is better -> ((baseline - optimized) / baseline) * 100
        lat_imp = ((baseline_lat - best_lat) / baseline_lat) * 100.0 if baseline_lat > 0 else 0.0
        # Throughput improvement: higher is better -> ((optimized - baseline) / baseline) * 100
        tps_imp = ((best_tps - baseline_tps) / baseline_tps) * 100.0 if baseline_tps > 0 else 0.0
        
        end_time = time.time()
        
        res = OptimizationResult(
            experiment_id=experiment_id,
            baseline=ref_baseline,
            configurations_tested=len(results),
            results=results,
            best_configuration=best_cfg,
            pareto_configurations=pareto_cfgs,
            improvement_vs_baseline={
                "latency_pct": lat_imp,
                "throughput_pct": tps_imp,
                "memory_pct": None # Memory measurement deferred until native profiling
            },
            execution_time_s=end_time - start_time
        )
        
        # Save to disk
        self._save_result(res)
        
        self.jobs[experiment_id]["status"] = "completed"
        self.jobs[experiment_id]["result"] = res.dict()
        
        # Restore baseline model state
        self.inference_engine.unload_model()
        config.MODEL_THREADS = original_threads
        self.inference_engine.load_model()
        
        return res
        
    def _save_result(self, res: OptimizationResult):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_{timestamp}_{res.experiment_id}.json"
        path = os.path.join(self.results_dir, filename)
        
        with open(path, 'w') as f:
            json.dump(res.dict(), f, indent=2)
        print(f"\nOptimization experiment saved to {path}")
