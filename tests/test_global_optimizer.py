import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from backend.api.main import app
from backend.optimizer.models import (
    OptimizationDimension, 
    OptimizationConfig, 
    Objective, 
    OptimizationRequest,
    OptimizationResult
)
from backend.optimizer.config_generator import ConfigurationGenerator
from backend.optimizer.scoring import ScoringEngine
from backend.optimizer.recommender import RecommendationEngine
from backend.optimizer.registry import ExperimentRegistry
from backend.optimizer.comparison import calculate_improvement, compare_configurations

client = TestClient(app)

def test_36_default_configurations_cartesian_product():
    gen = ConfigurationGenerator()
    configs = gen.generate_configurations(dimension=OptimizationDimension.GLOBAL)
    
    # 1. Exactly 36 configurations
    assert len(configs) == 36
    
    # 2. No duplicates
    ids = [c.dict()["configuration_id"] for c in configs]
    assert len(ids) == len(set(ids))
    
    # 3. Correct Cartesian product components
    quants = sorted(list(set(c.quantization for c in configs)))
    threads = sorted(list(set(c.threads for c in configs)))
    contexts = sorted(list(set(c.context_size for c in configs)))
    
    assert quants == ["Q4_K_M", "Q5_K_M", "Q8_0"]
    assert threads == [2, 4, 6, 8]
    assert contexts == [1024, 2048, 4096]

def test_configuration_ids_format():
    gen = ConfigurationGenerator()
    configs = gen.generate_configurations(dimension=OptimizationDimension.GLOBAL)
    
    for c in configs:
        expected_id = f"cfg_{c.quantization}_T{c.threads}_C{c.context_size}"
        assert c.dict()["configuration_id"] == expected_id

def test_invalid_parameter_rejections():
    # Invalid quantization
    with pytest.raises(ValueError):
        OptimizationConfig(quantization="INVALID_Q", threads=4, context_size=2048)
        
    # Invalid negative/zero threads
    with pytest.raises(ValueError):
        OptimizationConfig(quantization="Q4_K_M", threads=0, context_size=2048)
        
    # Invalid negative/zero context
    with pytest.raises(ValueError):
        OptimizationConfig(quantization="Q4_K_M", threads=4, context_size=-1024)
        
    # Invalid batch size
    with pytest.raises(ValueError):
        OptimizationConfig(quantization="Q4_K_M", threads=4, context_size=2048, batch_size=4)

def test_search_space_metadata():
    gen = ConfigurationGenerator()
    meta = gen.get_search_space_metadata()
    
    assert meta["quantizations"] == ["Q4_K_M", "Q5_K_M", "Q8_0"]
    assert meta["threads"] == [2, 4, 6, 8]
    assert meta["contexts"] == [1024, 2048, 4096]
    assert meta["total_configurations"] == 36

def test_api_search_space_endpoint():
    res = client.get("/api/optimization/search-space")
    assert res.status_code == 200
    data = res.json()
    assert data["dimension"] == "global"
    assert data["total_configurations"] == 36
    assert len(data["quantizations"]) == 3
    assert len(data["threads"]) == 4
    assert len(data["contexts"]) == 3

def test_3d_pareto_dominance_logic():
    # Setup test matrix
    results = [
        {
            "configuration_id": "cfg_Q8_0_T6_C4096",
            "configuration": {"quantization": "Q8_0", "threads": 6, "context_size": 4096},
            "results": {"mean_latency_ms": 2000.0, "mean_tokens_per_second": 45.0, "model_size_mb": 644.41}
        },
        {
            "configuration_id": "cfg_Q4_K_M_T6_C4096",
            "configuration": {"quantization": "Q4_K_M", "threads": 6, "context_size": 4096},
            "results": {"mean_latency_ms": 2500.0, "mean_tokens_per_second": 36.0, "model_size_mb": 468.64}
        },
        {
            "configuration_id": "cfg_Q8_0_T2_C1024",
            "configuration": {"quantization": "Q8_0", "threads": 2, "context_size": 1024},
            "results": {"mean_latency_ms": 3200.0, "mean_tokens_per_second": 28.0, "model_size_mb": 644.41}
        }
    ]
    
    pareto = ScoringEngine.get_pareto_frontier(results)
    
    # Q8_0_T6_C4096 dominates Q8_0_T2_C1024 (same size, but strictly lower latency and strictly higher throughput)
    # Q4_K_M_T6_C4096 is Pareto-optimal because it has smaller model size (468MB vs 644MB)
    assert len(pareto) == 2
    pareto_ids = [p["configuration_id"] for p in pareto]
    assert "cfg_Q8_0_T6_C4096" in pareto_ids
    assert "cfg_Q4_K_M_T6_C4096" in pareto_ids
    assert "cfg_Q8_0_T2_C1024" not in pareto_ids

def test_all_four_objectives_scoring():
    results = [
        {
            "configuration_id": "cfg_fast",
            "configuration": {"quantization": "Q8_0", "threads": 6, "context_size": 4096},
            "results": {"mean_latency_ms": 1800.0, "mean_tokens_per_second": 42.0, "model_size_mb": 644.41}
        },
        {
            "configuration_id": "cfg_high_tps",
            "configuration": {"quantization": "Q8_0", "threads": 8, "context_size": 2048},
            "results": {"mean_latency_ms": 2100.0, "mean_tokens_per_second": 50.0, "model_size_mb": 644.41}
        },
        {
            "configuration_id": "cfg_small_size",
            "configuration": {"quantization": "Q4_K_M", "threads": 4, "context_size": 1024},
            "results": {"mean_latency_ms": 2600.0, "mean_tokens_per_second": 32.0, "model_size_mb": 468.64}
        }
    ]
    
    # 1. SPEED objective -> cfg_fast wins
    best_speed = ScoringEngine.score_results(results, Objective.SPEED)
    assert best_speed["configuration_id"] == "cfg_fast"
    
    # 2. THROUGHPUT objective -> cfg_high_tps wins
    best_tps = ScoringEngine.score_results(results, Objective.THROUGHPUT)
    assert best_tps["configuration_id"] == "cfg_high_tps"
    
    # 3. SIZE objective -> cfg_small_size wins
    best_size = ScoringEngine.score_results(results, Objective.SIZE)
    assert best_size["configuration_id"] == "cfg_small_size"
    
    # 4. BALANCED objective -> calculates score cleanly without crashing
    best_bal = ScoringEngine.score_results(results, Objective.BALANCED)
    assert best_bal is not None

def test_recommender_payload_and_evidence():
    mock_reg = MagicMock()
    mock_reg.get_latest_experiment.return_value = {
        "experiment_id": "exp_global_1",
        "workload_type": "short_generation",
        "dimension": "global",
        "platform": {"provider": "local", "architecture": "amd64"},
        "baseline": {"results": {"mean_latency_ms": 2671.42, "mean_tokens_per_second": 32.73}},
        "results": [
            {
                "configuration_id": "cfg_Q8_0_T6_C4096",
                "configuration": {"quantization": "Q8_0", "threads": 6, "context_size": 4096},
                "results": {"mean_latency_ms": 1950.0, "mean_tokens_per_second": 45.0, "model_size_mb": 644.41}
            },
            {
                "configuration_id": "cfg_Q4_K_M_T4_C2048",
                "configuration": {"quantization": "Q4_K_M", "threads": 4, "context_size": 2048},
                "results": {"mean_latency_ms": 2200.0, "mean_tokens_per_second": 40.0, "model_size_mb": 468.64}
            }
        ]
    }
    
    rec_engine = RecommendationEngine(mock_reg)
    rec = rec_engine.recommend(workload="short_generation", objective="speed")
    
    assert rec["status"] == "success"
    assert rec["configuration_id"] == "cfg_Q8_0_T6_C4096"
    assert rec["recommended_configuration"]["quantization"] == "Q8_0"
    assert len(rec["evidence"]) >= 3
    assert "quality_score: null" in rec["evidence"][-1]
    assert rec["baseline_improvement"]["latency_pct"] > 0

def test_historical_registry_discovery():
    registry = ExperimentRegistry()
    experiments = registry.list_experiments()
    assert len(experiments) >= 2
    exp_ids = [e["experiment_id"] for e in experiments]
    assert len(exp_ids) == len(set(exp_ids)) # No duplicate entries

def test_comparison_calculations():
    base = {"results": {"mean_latency_ms": 2000.0, "mean_tokens_per_second": 30.0, "model_size_mb": 600.0}}
    cand = {"results": {"mean_latency_ms": 1500.0, "mean_tokens_per_second": 45.0, "model_size_mb": 450.0}}
    
    comp = compare_configurations(base, cand)
    assert comp["improvements"]["latency_pct"] == 25.0
    assert comp["improvements"]["throughput_pct"] == 50.0
    assert comp["improvements"]["size_reduction_pct"] == 25.0
