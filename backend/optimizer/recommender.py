from typing import Dict, Any, Optional, List
from .models import Objective
from .scoring import ScoringEngine
from .registry import ExperimentRegistry
from .comparison import calculate_improvement

class RecommendationEngine:
    """Provides objective-driven configuration recommendations from real experiment measurements."""

    def __init__(self, registry: Optional[ExperimentRegistry] = None):
        self.registry = registry or ExperimentRegistry()
        self.scoring_engine = ScoringEngine()

    def recommend(
        self, 
        workload: str = "short_generation", 
        objective: str = "speed",
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Determines the optimal configuration for a given workload and optimization objective.
        Returns the recommendation along with Pareto context, metrics, and baseline improvement.
        """
        obj_enum = Objective(objective.lower())
        experiment = self.registry.get_latest_experiment(workload_type=workload)
        
        if not experiment or "results" not in experiment or not experiment["results"]:
            return {
                "status": "no_data",
                "workload": workload,
                "objective": objective,
                "message": f"No optimization experiment found for workload '{workload}'. Run an optimization sweep first."
            }
            
        results = [dict(r) for r in experiment["results"]]
        
        # Apply optional constraints (e.g. max_threads, max_model_size_mb)
        if constraints:
            if "max_threads" in constraints:
                results = [r for r in results if r["configuration"]["threads"] <= constraints["max_threads"]]
            if "max_model_size_mb" in constraints:
                results = [r for r in results if r["results"].get("model_size_mb", 0.0) <= constraints["max_model_size_mb"]]
            if "quantization" in constraints:
                results = [r for r in results if r["configuration"].get("quantization") == constraints["quantization"].upper()]
                
        if not results:
            return {
                "status": "no_matching_configuration",
                "workload": workload,
                "objective": objective,
                "message": "No configuration matched the specified constraints."
            }

        best_cfg = self.scoring_engine.score_results(results, obj_enum)
        pareto_cfgs = self.scoring_engine.get_pareto_frontier(results)
        
        cfg = best_cfg["configuration"]
        res = best_cfg["results"]
        mean_lat = res["mean_latency_ms"]
        mean_tps = res["mean_tokens_per_second"]
        model_size = res.get("model_size_mb") or best_cfg.get("model_size_mb", 0.0)
        load_time = res.get("load_time_ms")
        quant = cfg.get("quantization", "Q4_K_M")
        th = cfg.get("threads")
        ctx = cfg.get("context_size")
        cfg_id = best_cfg.get("configuration_id") or cfg.get("configuration_id") or f"cfg_{quant}_T{th}_C{ctx}"
        
        # Build transparent, evidence-based explanation
        base_res = experiment.get("baseline", {}).get("results", {})
        base_lat = base_res.get("mean_latency_ms", 0.0)
        base_tps = base_res.get("mean_tokens_per_second", 0.0)
        
        lat_imp = calculate_improvement(base_lat, mean_lat, lower_is_better=True) if base_lat > 0 else 0.0
        tps_imp = calculate_improvement(base_tps, mean_tps, lower_is_better=False) if base_tps > 0 else 0.0
        
        evidence = []
        if lat_imp > 0:
            evidence.append(f"{lat_imp:.1f}% lower latency than baseline ({mean_lat:.1f}ms vs {base_lat:.1f}ms)")
        elif lat_imp < 0:
            evidence.append(f"{-lat_imp:.1f}% higher latency than baseline ({mean_lat:.1f}ms vs {base_lat:.1f}ms)")
            
        if tps_imp > 0:
            evidence.append(f"{tps_imp:.1f}% higher throughput than baseline ({mean_tps:.1f} tok/s vs {base_tps:.1f} tok/s)")
            
        evidence.append(f"{model_size:.1f} MB model footprint ({quant})")
        if best_cfg.get("pareto_optimal", True):
            evidence.append("Pareto-optimal trade-off across (latency, throughput, model size)")
        evidence.append("Model quality score is unmeasured / quality_score: null")

        # Tailor concise human-readable reason
        if obj_enum == Objective.SPEED:
            reason = f"Delivers the lowest measured mean latency ({mean_lat:.1f} ms) with {mean_tps:.1f} tok/s throughput on {experiment.get('platform', {}).get('architecture', 'host')}."
        elif obj_enum == Objective.THROUGHPUT:
            reason = f"Delivers the highest measured generation throughput ({mean_tps:.1f} tok/s) on {experiment.get('platform', {}).get('architecture', 'host')}."
        elif obj_enum == Objective.SIZE:
            reason = f"Provides the smallest model footprint ({model_size:.1f} MB) while maintaining {mean_tps:.1f} tok/s throughput."
        else:
            reason = f"Provides the optimal balanced trade-off between latency ({mean_lat:.1f} ms) and throughput ({mean_tps:.1f} tok/s)."

        return {
            "status": "success",
            "experiment_id": experiment.get("experiment_id"),
            "workload": workload,
            "objective": objective,
            "dimension": experiment.get("dimension", "global"),
            "platform": experiment.get("platform", {}),
            "configuration_id": cfg_id,
            "recommended_configuration": {
                "configuration_id": cfg_id,
                "quantization": quant,
                "threads": th,
                "context_size": ctx,
                "batch_size": cfg.get("batch_size", 1)
            },
            "metrics": {
                "mean_latency_ms": mean_lat,
                "median_latency_ms": res.get("median_latency_ms"),
                "p95_latency_ms": res.get("p95_latency_ms"),
                "std_latency_ms": res.get("std_latency_ms"),
                "mean_tokens_per_second": mean_tps,
                "model_size_mb": model_size,
                "load_time_ms": load_time,
                "quality_score": None,
                "memory_mb": res.get("memory_mb")
            },
            "score": round(best_cfg.get("score", 1.0), 4),
            "pareto_optimal": best_cfg.get("pareto_optimal", True),
            "baseline_improvement": {
                "latency_pct": lat_imp,
                "throughput_pct": tps_imp
            },
            "evidence": evidence,
            "reason": reason,
            "pareto_configurations": [
                {
                    "configuration_id": p.get("configuration_id") or f"cfg_{p['configuration'].get('quantization', 'Q4_K_M')}_T{p['configuration']['threads']}_C{p['configuration']['context_size']}",
                    "quantization": p["configuration"].get("quantization", "Q4_K_M"),
                    "threads": p["configuration"]["threads"],
                    "context_size": p["configuration"]["context_size"],
                    "model_size_mb": p["results"].get("model_size_mb", 0.0),
                    "mean_latency_ms": p["results"]["mean_latency_ms"],
                    "mean_tokens_per_second": p["results"]["mean_tokens_per_second"]
                }
                for p in pareto_cfgs
            ]
        }
