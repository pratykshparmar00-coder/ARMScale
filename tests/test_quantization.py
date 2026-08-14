import os
import pytest
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

def test_variant_catalog():
    assert "Q4_K_M" in AVAILABLE_VARIANTS
    assert "Q5_K_M" in AVAILABLE_VARIANTS
    assert "Q8_0" in AVAILABLE_VARIANTS
    
    q4 = AVAILABLE_VARIANTS["Q4_K_M"]
    assert q4.filename == "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    assert q4.expected_bytes == 491400032
    assert q4.license == "Apache-2.0"
    assert q4.quality_score is None

def test_variant_identity_and_path():
    path_q4 = get_model_path_for_variant("Q4_K_M")
    assert path_q4.endswith("qwen2.5-0.5b-instruct-q4_k_m.gguf")
    
    with pytest.raises(ValueError):
        get_model_path_for_variant("NON_EXISTENT_QUANT")

def test_downloaded_status():
    assert is_variant_downloaded("Q4_K_M") is True
    assert is_variant_downloaded("Q5_K_M") is True
    assert is_variant_downloaded("Q8_0") is True

def test_sha256_checksum():
    q4_path = get_model_path_for_variant("Q4_K_M")
    sha = calculate_file_sha256(q4_path)
    assert sha == "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"

def test_quantization_config_generator():
    gen = ConfigurationGenerator()
    configs = gen.generate_configurations(
        dimension=OptimizationDimension.QUANTIZATION,
        override_quantizations=["Q4_K_M", "Q5_K_M", "Q8_0"],
        fixed_thread_count=6,
        fixed_context_size=4096
    )
    assert len(configs) == 3
    assert configs[0].quantization == "Q4_K_M"
    assert configs[0].threads == 6
    assert configs[0].context_size == 4096
    assert configs[1].quantization == "Q5_K_M"
    assert configs[2].quantization == "Q8_0"

def test_quantization_pareto_scoring():
    # Test Pareto dominance across (latency, throughput, model_size)
    results = [
        {
            "configuration": {"quantization": "Q4_K_M", "threads": 6, "context_size": 4096},
            "results": {"mean_latency_ms": 2000.0, "mean_tokens_per_second": 45.0, "model_size_mb": 468.64}
        },
        {
            "configuration": {"quantization": "Q5_K_M", "threads": 6, "context_size": 4096},
            "results": {"mean_latency_ms": 2050.0, "mean_tokens_per_second": 44.0, "model_size_mb": 498.00}
        },
        {
            "configuration": {"quantization": "Q8_0", "threads": 6, "context_size": 4096},
            "results": {"mean_latency_ms": 2200.0, "mean_tokens_per_second": 42.0, "model_size_mb": 644.41}
        }
    ]
    
    # Q4_K_M has lower latency, higher throughput, and smaller size -> Strictly dominates Q5_K_M and Q8_0
    pareto = ScoringEngine.get_pareto_frontier(results)
    assert len(pareto) == 1
    assert pareto[0]["configuration"]["quantization"] == "Q4_K_M"
    assert results[0]["pareto_optimal"] is True
    assert results[1]["pareto_optimal"] is False
    assert results[2]["pareto_optimal"] is False
