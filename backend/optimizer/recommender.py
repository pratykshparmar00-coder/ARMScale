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
        
        # Build transparent explanation
        if obj_enum == Objective.SPEED:
            reason = f"Achieved the lowest mean generation latency ({mean_lat:.2f} ms) across all tested configurations."
        elif obj_enum == Objective.THROUGHPUT:
            reason = f"Achieved the highest token throughput ({mean_tps:.2f} tok/s) across all tested configurations."
        elif obj_enum == Objective.BALANCED:
            reason = f"Optimal balanced score ({best_cfg.get('score', 0):.4f}) combining 50% normalized latency and 50% normalized throughput."
        else:
            reason = f"Highest score under objective '{objective}'."

        # Compute improvements against reference baseline in experiment
        base_res = experiment.get("baseline", {}).get("results", {})
        base_lat = base_res.get("mean_latency_ms", 0.0)
        base_tps = base_res.get("mean_tokens_per_second", 0.0)
        
        lat_imp = calculate_improvement(base_lat, mean_lat, lower_is_better=True) if base_lat > 0 else 0.0
        tps_imp = calculate_improvement(base_tps, mean_tps, lower_is_better=False) if base_tps > 0 else 0.0
        
        return {
            "status": "success",
            "experiment_id": experiment.get("experiment_id"),
            "workload": workload,
            "objective": objective,
            "platform": experiment.get("platform", {}),
            "recommended_configuration": {
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
                "memory_mb": res.get("memory_mb")
            },
            "baseline_improvement": {
                "latency_pct": lat_imp,
                "throughput_pct": tps_imp
            },
            "pareto_configurations": [
                {
                    "threads": p["configuration"]["threads"],
                    "context_size": p["configuration"]["context_size"],
                    "mean_latency_ms": p["results"]["mean_latency_ms"],
                    "mean_tokens_per_second": p["results"]["mean_tokens_per_second"]
                }
                for p in pareto_cfgs
            ],
            "reason": reason
        }
