import os
import time
import uuid
import json
import glob
import datetime
from typing import Dict, Any, List, Optional

from .models import OptimizationRequest, OptimizationResult, OptimizationConfig, Objective, OptimizationDimension
from .config_generator import ConfigurationGenerator
from .scoring import ScoringEngine
from backend.benchmark.engine import BenchmarkEngine, WorkloadType
from backend.inference.llama_cpp_engine import LlamaCppEngine
from backend.platform.detector import get_platform
from backend.config import config

class OptimizationEngine:
    def __init__(self, inference_engine: LlamaCppEngine, benchmark_engine: BenchmarkEngine):
        self.inference_engine = inference_engine
        self.benchmark_engine = benchmark_engine
        self.config_generator = ConfigurationGenerator()
        self.scoring_engine = ScoringEngine()
        
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.results_dir = os.path.join(self.root_dir, "benchmarks", "results", "optimization")
        self.global_results_dir = os.path.join(self.results_dir, "global")
        
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        if not os.path.exists(self.global_results_dir):
            os.makedirs(self.global_results_dir)
            
        self.jobs = {}

    def get_reference_baseline(self, workload_type: str = "short_generation") -> Optional[Dict[str, Any]]:
        """Finds the canonical baseline benchmark recorded for the specified workload."""
        baseline_dir = os.path.join(self.root_dir, "benchmarks", "results")
        json_files = sorted(glob.glob(os.path.join(baseline_dir, "benchmark_*.json")))
        for fpath in json_files:
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    if data.get("model") != "mock/mock" and data.get("runtime") != "mock":
                        w_type = data.get("workload", {}).get("type", "short_generation")
                        if w_type == workload_type:
                            return data
            except Exception as e:
                print(f"Warning: Could not read baseline candidate {fpath}: {e}")
        return None

    def get_latest_optimization(self, workload_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Finds the latest completed optimization experiment JSON across all experiment folders."""
        all_json_files = glob.glob(os.path.join(self.results_dir, "*.json")) + glob.glob(os.path.join(self.global_results_dir, "*.json"))
        sorted_files = sorted(all_json_files, key=os.path.getmtime)
        
        if not sorted_files:
            return None
            
        if not workload_type:
            try:
                with open(sorted_files[-1], 'r') as f:
                    return json.load(f)
            except Exception:
                return None
                
        for fpath in reversed(sorted_files):
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    if data.get("workload_type") == workload_type:
                        return data
            except Exception:
                continue
        return None

    def start_optimization(self, request: OptimizationRequest) -> str:
        experiment_id = str(uuid.uuid4())[:8]
        self.jobs[experiment_id] = {
            "status": "queued",
            "dimension": request.dimension.value if isinstance(request.dimension, OptimizationDimension) else str(request.dimension),
            "workload_type": request.workload_type,
            "completed": 0,
            "total": 0,
            "current_configuration": None,
            "result": None
        }
        return experiment_id

    def run_optimization_sync(self, request: OptimizationRequest, experiment_id: str) -> OptimizationResult:
        start_time = time.time()
        self.jobs[experiment_id]["status"] = "running"
        
        dim = request.dimension if isinstance(request.dimension, OptimizationDimension) else OptimizationDimension(request.dimension)
        w_type = WorkloadType(request.workload_type)
        
        # 1. Retrieve or Run Baseline for this workload
        ref_baseline = self.get_reference_baseline(request.workload_type)
        original_threads = config.MODEL_THREADS
        original_context = config.MODEL_CONTEXT_SIZE
        original_quant = "Q4_K_M"
        
        if not ref_baseline:
            print(f"No existing baseline found for workload '{w_type.value}'. Running initial baseline...")
            self.inference_engine.load_model(threads=original_threads, context_size=original_context, quantization=original_quant)
            ref_baseline = self.benchmark_engine.run_baseline(
                threads=original_threads, 
                context_size=original_context,
                quantization=original_quant,
                workload_type=w_type, 
                save=True
            )
        else:
            print(f"Referencing recorded baseline for '{w_type.value}' (Mean Latency: {ref_baseline['results']['mean_latency_ms']:.2f}ms, Mean TPS: {ref_baseline['results']['mean_tokens_per_second']:.2f})")

        # 2. Generate Candidate Configurations
        configs = self.config_generator.generate_configurations(
            dimension=dim,
            override_threads=request.threads_to_test,
            override_contexts=request.context_sizes_to_test,
            override_quantizations=request.quantizations_to_test,
            fixed_thread_count=6,
            fixed_context_size=4096,
            fixed_quantization="Q4_K_M"
        )
        search_space_meta = self.config_generator.get_search_space_metadata(
            override_quantizations=request.quantizations_to_test,
            override_threads=request.threads_to_test,
            override_contexts=request.context_sizes_to_test
        )
        self.jobs[experiment_id]["total"] = len(configs)
        
        results = []
        
        # 3. Benchmark Each Candidate Configuration Sequentially
        for idx, cfg in enumerate(configs):
            self.jobs[experiment_id]["completed"] = idx
            self.jobs[experiment_id]["current_configuration"] = cfg.dict()
            cfg_id = cfg.configuration_id or f"cfg_{cfg.quantization}_T{cfg.threads}_C{cfg.context_size}"
            
            print(f"\n[{idx+1}/{len(configs)}] Testing {cfg_id} (Quant={cfg.quantization}, Threads={cfg.threads}, Context={cfg.context_size})")
            
            # Safely unload previous model before switching variants to avoid memory/thread contention
            self.inference_engine.unload_model()
            success = self.inference_engine.load_model(
                threads=cfg.threads, 
                context_size=cfg.context_size,
                quantization=cfg.quantization
            )
            if not success:
                print(f"Failed to load model for configuration {cfg.dict()}, skipping.")
                continue
                
            # Run benchmark under exact workload (2 warmups, 5 measured runs)
            bench_res = self.benchmark_engine.run_baseline(
                threads=cfg.threads,
                context_size=cfg.context_size,
                quantization=cfg.quantization,
                workload_type=w_type,
                save=True
            )
            bench_res['configuration_id'] = cfg_id
            bench_res['configuration']['configuration_id'] = cfg_id
            bench_res['configuration']['threads'] = cfg.threads
            bench_res['configuration']['context_size'] = cfg.context_size
            bench_res['configuration']['quantization'] = cfg.quantization
            results.append(bench_res)
            
        # 4. Score and Analyze Results
        best_cfg = self.scoring_engine.score_results(results, request.objective)
        pareto_cfgs = self.scoring_engine.get_pareto_frontier(results)
        
        # 5. Calculate Improvements vs Reference Baseline
        baseline_lat = ref_baseline['results']['mean_latency_ms']
        baseline_tps = ref_baseline['results']['mean_tokens_per_second']
        
        best_lat = best_cfg['results']['mean_latency_ms'] if best_cfg else baseline_lat
        best_tps = best_cfg['results']['mean_tokens_per_second'] if best_cfg else baseline_tps
        
        lat_imp = ((baseline_lat - best_lat) / baseline_lat) * 100.0 if baseline_lat > 0 else 0.0
        tps_imp = ((best_tps - baseline_tps) / baseline_tps) * 100.0 if baseline_tps > 0 else 0.0
        
        end_time = time.time()
        platform_info = get_platform().to_dict()
        
        res = OptimizationResult(
            experiment_id=experiment_id,
            dimension=dim.value,
            workload_type=w_type.value,
            platform=platform_info,
            search_space=search_space_meta,
            baseline=ref_baseline,
            configurations_tested=len(results),
            results=results,
            best_configuration=best_cfg,
            pareto_configurations=pareto_cfgs,
            improvement_vs_baseline={
                "latency_pct": lat_imp,
                "throughput_pct": tps_imp,
                "memory_pct": None
            },
            execution_time_s=end_time - start_time
        )
        
        # Save to disk
        self._save_result(res)
        
        self.jobs[experiment_id]["status"] = "completed"
        self.jobs[experiment_id]["result"] = res.dict()
        
        # Restore baseline model state
        self.inference_engine.unload_model()
        self.inference_engine.load_model(threads=original_threads, context_size=original_context, quantization=original_quant)
        
        return res
        
    def _save_result(self, res: OptimizationResult):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = self.global_results_dir if res.dimension == "global" else self.results_dir
        filename = f"optimization_{res.dimension}_{res.workload_type}_{timestamp}_{res.experiment_id}.json"
        path = os.path.join(target_dir, filename)
        
        with open(path, 'w') as f:
            json.dump(res.dict(), f, indent=2)
        print(f"\nOptimization experiment saved to {path}")
