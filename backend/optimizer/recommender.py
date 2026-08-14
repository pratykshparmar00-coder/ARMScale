from typing import Dict, Any, Optional
from .models import Objective
from .scoring import ScoringEngine
from .registry import ExperimentRegistry
from .comparison import calculate_improvement

class RecommendationEngine:
    """Provides objective-driven configuration recommendations from real experiment measurements."""

    def __init__(self, registry: Optional[ExperimentRegistry] = None):
        self.registry = registry or ExperimentRegistry()
        self.scoring_engine = ScoringEngine()

    def recommend(self, workload: str = "short_generation", objective: str = "speed") -> Dict[str, Any]:
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
        best_cfg = self.scoring_engine.score_results(results, obj_enum)
        pareto_cfgs = self.scoring_engine.get_pareto_frontier(results)
        
        cfg = best_cfg["configuration"]
        res = best_cfg["results"]
        mean_lat = res["mean_latency_ms"]
        mean_tps = res["mean_tokens_per_second"]
        model_size = res.get("model_size_mb") or best_cfg.get("model_size_mb", 0.0)
        quant = cfg.get("quantization", "Q4_K_M")
        
        # Build transparent, evidence-based explanation
        base_res = experiment.get("baseline", {}).get("results", {})
        base_lat = base_res.get("mean_latency_ms", 0.0)
        base_tps = base_res.get("mean_tokens_per_second", 0.0)
        
        lat_imp = calculate_improvement(base_lat, mean_lat, lower_is_better=True) if base_lat > 0 else 0.0
        tps_imp = calculate_improvement(base_tps, mean_tps, lower_is_better=False) if base_tps > 0 else 0.0
        
        reasons = []
        if lat_imp > 0:
            reasons.append(f"{lat_imp:.1f}% lower latency than baseline")
        elif lat_imp < 0:
            reasons.append(f"{-lat_imp:.1f}% higher latency than baseline")
            
        if tps_imp > 0:
            reasons.append(f"{tps_imp:.1f}% higher throughput than baseline")
            
        reasons.append(f"{model_size:.1f} MB model footprint ({quant})")
        if best_cfg.get("pareto_optimal", True):
            reasons.append("Pareto-optimal trade-off")
        reasons.append("model quality score unavailable (unmeasured)")

        reason_str = f"Selected under '{objective.upper()}' objective: " + "; ".join(reasons) + "."
        
        return {
            "status": "success",
            "experiment_id": experiment.get("experiment_id"),
            "workload": workload,
            "objective": objective,
            "dimension": experiment.get("dimension", "unknown"),
            "platform": experiment.get("platform", {}),
            "recommended_configuration": {
                "quantization": quant,
                "threads": cfg.get("threads"),
                "context_size": cfg.get("context_size"),
                "batch_size": cfg.get("batch_size", 1)
            },
            "metrics": {
                "mean_latency_ms": mean_lat,
                "median_latency_ms": res.get("median_latency_ms"),
                "p95_latency_ms": res.get("p95_latency_ms"),
                "std_latency_ms": res.get("std_latency_ms"),
                "mean_tokens_per_second": mean_tps,
                "model_size_mb": model_size,
                "load_time_ms": res.get("load_time_ms"),
                "quality_score": None,
                "memory_mb": res.get("memory_mb")
            },
            "baseline_improvement": {
                "latency_pct": lat_imp,
                "throughput_pct": tps_imp
            },
            "pareto_configurations": [
                {
                    "quantization": p["configuration"].get("quantization", "Q4_K_M"),
                    "threads": p["configuration"]["threads"],
                    "context_size": p["configuration"]["context_size"],
                    "model_size_mb": p["results"].get("model_size_mb", 0.0),
                    "mean_latency_ms": p["results"]["mean_latency_ms"],
                    "mean_tokens_per_second": p["results"]["mean_tokens_per_second"]
                }
                for p in pareto_cfgs
            ],
            "reason": reason_str
        }
