import pytest
from backend.optimizer.config_generator import ConfigurationGenerator
from backend.optimizer.scoring import ScoringEngine
from backend.optimizer.models import Objective, OptimizationConfig

def test_config_generation():
    generator = ConfigurationGenerator()
    # Mocking physical cores to a known value for deterministic testing
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
        {"results": {"mean_latency_ms": 100, "mean_tokens_per_second": 10}, "configuration": {"threads": 1}},
        {"results": {"mean_latency_ms": 50, "mean_tokens_per_second": 20}, "configuration": {"threads": 2}},
        {"results": {"mean_latency_ms": 200, "mean_tokens_per_second": 5}, "configuration": {"threads": 4}}
    ]
    
    best_speed = ScoringEngine.score_results(results.copy(), Objective.SPEED)
    assert best_speed['configuration']['threads'] == 2 # 50ms is fastest
    
    best_tps = ScoringEngine.score_results(results.copy(), Objective.THROUGHPUT)
    assert best_tps['configuration']['threads'] == 2 # 20 is highest tps

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
