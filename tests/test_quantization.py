import os
import pytest
from fastapi.testclient import TestClient
from backend.api.main import app
from backend.inference.models import (
    AVAILABLE_VARIANTS,
    get_model_path_for_variant,
    is_variant_downloaded,
    get_variant_identity,
    calculate_file_sha256
)
from backend.optimizer.models import OptimizationDimension, OptimizationConfig, Objective
from backend.optimizer.config_generator import ConfigurationGenerator
from backend.optimizer.scoring import ScoringEngine
from backend.optimizer.recommender import RecommendationEngine
from unittest.mock import MagicMock

client = TestClient(app)

def test_variant_catalog():
    assert "Q4_K_M" in AVAILABLE_VARIANTS
    assert "Q5_K_M" in AVAILABLE_VARIANTS
    assert "Q8_0" in AVAILABLE_VARIANTS
    
    q4 = AVAILABLE_VARIANTS["Q4_K_M"]
    assert q4.filename == "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    assert q4.expected_bytes == 491400032
    assert q4.license == "Apache-2.0"
    assert q4.quality_score is None

    q5 = AVAILABLE_VARIANTS["Q5_K_M"]
    assert q5.filename == "qwen2.5-0.5b-instruct-q5_k_m.gguf"
    assert q5.expected_bytes == 522186592

    q8 = AVAILABLE_VARIANTS["Q8_0"]
    assert q8.filename == "qwen2.5-0.5b-instruct-q8_0.gguf"
    assert q8.expected_bytes == 675710816

def test_filename_mapping_and_rejection():
    path_q4 = get_model_path_for_variant("Q4_K_M")
    assert path_q4.endswith("qwen2.5-0.5b-instruct-q4_k_m.gguf")
    
    path_q5 = get_model_path_for_variant("Q5_K_M")
    assert path_q5.endswith("qwen2.5-0.5b-instruct-q5_k_m.gguf")
    
    path_q8 = get_model_path_for_variant("Q8_0")
    assert path_q8.endswith("qwen2.5-0.5b-instruct-q8_0.gguf")
    
    with pytest.raises(ValueError):
        get_model_path_for_variant("INVALID_QUANT_FORMAT")

def test_downloaded_detection():
    assert is_variant_downloaded("Q4_K_M") is True
    assert is_variant_downloaded("Q5_K_M") is True
    assert is_variant_downloaded("Q8_0") is True

def test_sha256_checksums():
    q4_path = get_model_path_for_variant("Q4_K_M")
    assert calculate_file_sha256(q4_path) == "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
    
    q5_path = get_model_path_for_variant("Q5_K_M")
    assert calculate_file_sha256(q5_path) == "041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55"
    
    q8_path = get_model_path_for_variant("Q8_0")
    assert calculate_file_sha256(q8_path) == "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"

def test_candidate_generation_and_filtering():
    gen = ConfigurationGenerator()
    
    # 1. Full quantization matrix
    configs = gen.generate_configurations(
        dimension=OptimizationDimension.QUANTIZATION,
        override_quantizations=["Q4_K_M", "Q5_K_M", "Q8_0"],
        fixed_thread_count=6,
        fixed_context_size=4096
    )
    assert len(configs) == 3
    assert [c.quantization for c in configs] == ["Q4_K_M", "Q5_K_M", "Q8_0"]
    assert all(c.threads == 6 and c.context_size == 4096 for c in configs)
    
    # 2. Subset
    subset_configs = gen.generate_configurations(
        dimension=OptimizationDimension.QUANTIZATION,
        override_quantizations=["Q4_K_M", "Q8_0"],
        fixed_thread_count=6,
        fixed_context_size=4096
    )
    assert len(subset_configs) == 2
    assert [c.quantization for c in subset_configs] == ["Q4_K_M", "Q8_0"]
    
    # 3. Invalid variant rejection
    filtered_configs = gen.generate_configurations(
        dimension=OptimizationDimension.QUANTIZATION,
        override_quantizations=["Q4_K_M", "NON_EXISTENT"],
        fixed_thread_count=6,
        fixed_context_size=4096
    )
    assert len(filtered_configs) == 1
    assert filtered_configs[0].quantization == "Q4_K_M"

def test_size_objective_scoring():
    results = [
        {
            "configuration": {"quantization": "Q4_K_M", "threads": 6, "context_size": 4096},
            "results": {"mean_latency_ms": 2500.0, "mean_tokens_per_second": 35.0, "model_size_mb": 468.64}
        },
        {
            "configuration": {"quantization": "Q8_0", "threads": 6, "context_size": 4096},
            "results": {"mean_latency_ms": 2200.0, "mean_tokens_per_second": 39.0, "model_size_mb": 644.41}
        }
    ]
    # Under SIZE objective, Q4_K_M must win because it has smaller model size
    winner = ScoringEngine.score_results(results, Objective.SIZE)
    assert winner["configuration"]["quantization"] == "Q4_K_M"

def test_3d_pareto_scoring():
    results = [
        {
            "configuration": {"quantization": "Q4_K_M", "threads": 6, "context_size": 4096},
            "results": {"mean_latency_ms": 2900.0, "mean_tokens_per_second": 35.0, "model_size_mb": 468.64}
        },
        {
            "configuration": {"quantization": "Q5_K_M", "threads": 6, "context_size": 4096},
            "results": {"mean_latency_ms": 2400.0, "mean_tokens_per_second": 36.0, "model_size_mb": 498.00}
        },
        {
            "configuration": {"quantization": "Q8_0", "threads": 6, "context_size": 4096},
            "results": {"mean_latency_ms": 2200.0, "mean_tokens_per_second": 39.0, "model_size_mb": 644.41}
        }
    ]
    # Q8_0 has best speed, Q4_K_M has smallest size, Q5_K_M is intermediate -> All 3 are Pareto-optimal
    pareto = ScoringEngine.get_pareto_frontier(results)
    assert len(pareto) == 3
    assert all(r["pareto_optimal"] for r in results)

def test_recommender_with_quantization_and_reasons():
    mock_reg = MagicMock()
    mock_reg.get_latest_experiment.return_value = {
        "experiment_id": "exp_q123",
        "workload_type": "short_generation",
        "dimension": "quantization",
        "baseline": {"results": {"mean_latency_ms": 2671.42, "mean_tokens_per_second": 32.73}},
        "results": [
            {
                "configuration": {"quantization": "Q8_0", "threads": 6, "context_size": 4096},
                "results": {"mean_latency_ms": 2237.84, "mean_tokens_per_second": 38.84, "model_size_mb": 644.41}
            },
            {
                "configuration": {"quantization": "Q4_K_M", "threads": 6, "context_size": 4096},
                "results": {"mean_latency_ms": 2916.56, "mean_tokens_per_second": 35.60, "model_size_mb": 468.64}
            }
        ]
    }
    
    rec_engine = RecommendationEngine(mock_reg)
    rec = rec_engine.recommend("short_generation", "speed")
    assert rec["status"] == "success"
    assert rec["recommended_configuration"]["quantization"] == "Q8_0"
    assert "lower latency than baseline" in rec["reason"]
    assert "model quality score unavailable" in rec["reason"]

def test_api_models_variants_endpoint():
    response = client.get("/api/models/variants")
    assert response.status_code == 200
    data = response.json()
    assert "Q4_K_M" in data
    assert "Q5_K_M" in data
    assert "Q8_0" in data
    assert data["Q4_K_M"]["is_downloaded"] is True
    assert data["Q5_K_M"]["is_downloaded"] is True
    assert data["Q8_0"]["is_downloaded"] is True
