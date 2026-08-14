import pytest
from backend.optimizer.config_generator import ConfigurationGenerator
from backend.optimizer.scoring import ScoringEngine
from backend.optimizer.models import Objective, OptimizationConfig

def test_config_generation():
    generator = ConfigurationGenerator()
    generator.physical_cores = 8
    generator.logical_cores = 16
    
    candidates = generator.generate_thread_candidates()
    assert 1 in candidates
    assert 2 in candidates
    assert 8 in candidates
    assert 16 in candidates
    
    configs = generator.generate_configurations()
    assert len(configs) == len(candidates)
    assert isinstance(configs[0], OptimizationConfig)

def test_scoring_engine():
    results = [
        {"results": {"mean_latency_ms": 100, "mean_tokens_per_second": 10, "memory_mb": None}, "configuration": {"threads": 1}},
        {"results": {"mean_latency_ms": 50, "mean_tokens_per_second": 20, "memory_mb": None}, "configuration": {"threads": 2}},
        {"results": {"mean_latency_ms": 200, "mean_tokens_per_second": 5, "memory_mb": None}, "configuration": {"threads": 4}}
    ]
    
    best_speed = ScoringEngine.score_results(results.copy(), Objective.SPEED)
    assert best_speed['configuration']['threads'] == 2 # 50ms is fastest
    
    best_tps = ScoringEngine.score_results(results.copy(), Objective.THROUGHPUT)
    assert best_tps['configuration']['threads'] == 2 # 20 is highest tps
    
    best_balanced = ScoringEngine.score_results(results.copy(), Objective.BALANCED)
    assert best_balanced['configuration']['threads'] == 2 # Best in both
    
    # Memory fallback
    best_mem = ScoringEngine.score_results(results.copy(), Objective.MEMORY)
    assert best_mem['configuration']['threads'] == 2
    assert "scoring_note" in best_mem

def test_pareto_frontier():
    results = [
        {"results": {"mean_latency_ms": 50, "mean_tokens_per_second": 10}, "configuration": {"threads": 1}}, # Pareto
        {"results": {"mean_latency_ms": 100, "mean_tokens_per_second": 5}, "configuration": {"threads": 2}}, # Dominated by 1
        {"results": {"mean_latency_ms": 60, "mean_tokens_per_second": 20}, "configuration": {"threads": 4}}, # Pareto
    ]
    
    pareto = ScoringEngine.get_pareto_frontier(results)
    assert len(pareto) == 2
    threads = [p['configuration']['threads'] for p in pareto]
    assert 1 in threads
    assert 4 in threads
    assert 2 not in threads

def test_improvement_calculations():
    baseline_lat = 2671.42
    optimized_lat = 2019.23
    lat_imp = ((baseline_lat - optimized_lat) / baseline_lat) * 100
    assert round(lat_imp, 2) == 24.41

    baseline_tps = 32.73
    optimized_tps = 43.36
    tps_imp = ((optimized_tps - baseline_tps) / baseline_tps) * 100
    assert round(tps_imp, 2) == 32.48
