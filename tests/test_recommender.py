import pytest
from unittest.mock import MagicMock
from backend.optimizer.recommender import RecommendationEngine
from backend.optimizer.registry import ExperimentRegistry

def test_recommendation_speed_and_throughput():
    mock_registry = MagicMock(spec=ExperimentRegistry)
    mock_registry.get_latest_experiment.return_value = {
        "experiment_id": "test1234",
        "workload_type": "short_generation",
        "baseline": {
            "results": {"mean_latency_ms": 2500.0, "mean_tokens_per_second": 30.0}
        },
        "results": [
            {
                "configuration": {"threads": 4, "context_size": 2048},
                "results": {"mean_latency_ms": 1900.0, "mean_tokens_per_second": 40.0}
            },
            {
                "configuration": {"threads": 6, "context_size": 4096},
                "results": {"mean_latency_ms": 2100.0, "mean_tokens_per_second": 45.0}
            }
        ]
    }
    
    recommender = RecommendationEngine(registry=mock_registry)
    
    # Speed recommendation should select fastest latency (1900 ms)
    rec_speed = recommender.recommend(workload="short_generation", objective="speed")
    assert rec_speed["status"] == "success"
    assert rec_speed["recommended_configuration"]["threads"] == 4
    assert rec_speed["recommended_configuration"]["context_size"] == 2048
    assert rec_speed["metrics"]["mean_latency_ms"] == 1900.0
    
    # Throughput recommendation should select highest TPS (45.0 tok/s)
    rec_tps = recommender.recommend(workload="short_generation", objective="throughput")
    assert rec_tps["status"] == "success"
    assert rec_tps["recommended_configuration"]["threads"] == 6
    assert rec_tps["recommended_configuration"]["context_size"] == 4096
    assert rec_tps["metrics"]["mean_tokens_per_second"] == 45.0

def test_recommendation_no_data():
    mock_registry = MagicMock(spec=ExperimentRegistry)
    mock_registry.get_latest_experiment.return_value = None
    
    recommender = RecommendationEngine(registry=mock_registry)
    rec = recommender.recommend(workload="unknown_workload", objective="speed")
    assert rec["status"] == "no_data"
