import os
import time
import uuid
import json
import csv
import datetime
from typing import Dict, Any, List

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
        
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmarks", "results", "optimization")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        self.jobs = {}

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
        
        # 1. Establish Baseline
        # We assume the default config is baseline. 
        # But we must run it directly to ensure comparable measurements in this exact moment.
        # However, user said: "Do not alter or overwrite the existing baseline. Create a new optimization experiment system that references the baseline."
        # We will run a baseline measurement for this experiment.
        print("Running optimization experiment baseline...")
        original_threads = config.MODEL_THREADS
        baseline_result = self.benchmark_engine.run_baseline(threads=original_threads)
        
        # 2. Generate Configurations
        configs = self.config_generator.generate_configurations(request.threads_to_test)
        self.jobs[experiment_id]["total"] = len(configs)
        
        results = []
        
        # 3. Benchmark Candidates
        for idx, cfg in enumerate(configs):
            self.jobs[experiment_id]["completed"] = idx
            self.jobs[experiment_id]["current_configuration"] = cfg.dict()
            
            print(f"Testing configuration {idx+1}/{len(configs)}: {cfg.threads} threads")
            
            # Reconfigure engine
            # We must unload and reload the model to apply thread changes in llama.cpp safely
            self.inference_engine.unload_model()
            
            # Temporarily override config
            original_config_threads = config.MODEL_THREADS
            config.MODEL_THREADS = cfg.threads
            
            success = self.inference_engine.load_model()
            if not success:
                print(f"Failed to load model for {cfg.threads} threads, skipping.")
                config.MODEL_THREADS = original_config_threads
                continue
                
            # Run benchmark
            # Wait! The requirement says "Do not mix model initialization time into steady-state generation latency."
            # The benchmark_engine only times generation, so we're good.
            bench_res = self.benchmark_engine.run_baseline(threads=cfg.threads)
            
            # Attach the configuration to the result
            bench_res['configuration']['threads'] = cfg.threads
            results.append(bench_res)
            
            # Restore config
            config.MODEL_THREADS = original_config_threads
            
        # 4. Analyze Results
        best_cfg = self.scoring_engine.score_results(results, request.objective)
        pareto_cfgs = self.scoring_engine.get_pareto_frontier(results)
        
        # 5. Improvement Calculation vs Baseline
        baseline_lat = baseline_result['results']['mean_latency_ms']
        baseline_tps = baseline_result['results']['mean_tokens_per_second']
        
        best_lat = best_cfg['results']['mean_latency_ms']
        best_tps = best_cfg['results']['mean_tokens_per_second']
        
        lat_imp = (baseline_lat - best_lat) / baseline_lat * 100 if baseline_lat > 0 else 0.0
        tps_imp = (best_tps - baseline_tps) / baseline_tps * 100 if baseline_tps > 0 else 0.0
        
        end_time = time.time()
        
        res = OptimizationResult(
            experiment_id=experiment_id,
            baseline=baseline_result,
            configurations_tested=len(results),
            results=results,
            best_configuration=best_cfg,
            pareto_configurations=pareto_cfgs,
            improvement_vs_baseline={
                "latency_pct": lat_imp,
                "throughput_pct": tps_imp,
                "memory_pct": 0.0 # Placeholder
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
