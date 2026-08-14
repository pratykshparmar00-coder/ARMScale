from typing import Dict, Any, Optional

def calculate_improvement(baseline_val: float, optimized_val: float, lower_is_better: bool = True) -> float:
    """
    Calculates percentage improvement:
    - If lower is better (e.g. latency, size): ((baseline - optimized) / baseline) * 100
    - If higher is better (e.g. throughput): ((optimized - baseline) / baseline) * 100
    """
    if baseline_val <= 0:
        return 0.0
    if lower_is_better:
        return ((baseline_val - optimized_val) / baseline_val) * 100.0
    else:
        return ((optimized_val - baseline_val) / baseline_val) * 100.0

def compare_configurations(baseline_result: Dict[str, Any], candidate_result: Dict[str, Any]) -> Dict[str, Any]:
    """Compares a candidate configuration against a baseline benchmark across latency, throughput, and size."""
    base_res = baseline_result.get("results", baseline_result)
    cand_res = candidate_result.get("results", candidate_result)
    
    base_lat = base_res.get("mean_latency_ms", 0.0)
    cand_lat = cand_res.get("mean_latency_ms", 0.0)
    
    base_tps = base_res.get("mean_tokens_per_second", 0.0)
    cand_tps = cand_res.get("mean_tokens_per_second", 0.0)
    
    base_size = base_res.get("model_size_mb", 0.0) or baseline_result.get("model_size_mb", 0.0)
    cand_size = cand_res.get("model_size_mb", 0.0) or candidate_result.get("model_size_mb", 0.0)
    
    lat_imp = calculate_improvement(base_lat, cand_lat, lower_is_better=True)
    tps_imp = calculate_improvement(base_tps, cand_tps, lower_is_better=False)
    size_red = calculate_improvement(base_size, cand_size, lower_is_better=True) if base_size > 0 and cand_size > 0 else 0.0
    
    return {
        "baseline": {
            "mean_latency_ms": base_lat,
            "mean_tokens_per_second": base_tps,
            "model_size_mb": base_size,
            "configuration": baseline_result.get("configuration", {})
        },
        "candidate": {
            "mean_latency_ms": cand_lat,
            "mean_tokens_per_second": cand_tps,
            "model_size_mb": cand_size,
            "configuration": candidate_result.get("configuration", {})
        },
        "improvements": {
            "latency_pct": lat_imp,
            "throughput_pct": tps_imp,
            "size_reduction_pct": size_red,
            "memory_pct": None # Deferred
        }
    }
