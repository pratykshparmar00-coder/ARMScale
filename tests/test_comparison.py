import pytest
from backend.optimizer.comparison import calculate_improvement, compare_configurations

def test_calculate_improvement_latency():
    # Latency: lower is better -> ((2000 - 1500) / 2000) * 100 = 25%
    imp = calculate_improvement(2000.0, 1500.0, lower_is_better=True)
    assert imp == 25.0

    # Regression: ((2000 - 2500) / 2000) * 100 = -25%
    reg = calculate_improvement(2000.0, 2500.0, lower_is_better=True)
    assert reg == -25.0

def test_calculate_improvement_throughput():
    # Throughput: higher is better -> ((40 - 20) / 20) * 100 = 100%
    imp = calculate_improvement(20.0, 40.0, lower_is_better=False)
    assert imp == 100.0

    # Regression: ((15 - 20) / 20) * 100 = -25%
    reg = calculate_improvement(20.0, 15.0, lower_is_better=False)
    assert reg == -25.0

def test_compare_configurations():
    base = {
        "results": {"mean_latency_ms": 2500.0, "mean_tokens_per_second": 30.0},
        "configuration": {"threads": 4, "context_size": 2048}
    }
    cand = {
        "results": {"mean_latency_ms": 2000.0, "mean_tokens_per_second": 39.0},
        "configuration": {"threads": 6, "context_size": 2048}
    }
    
    comp = compare_configurations(base, cand)
    assert comp["improvements"]["latency_pct"] == 20.0
    assert comp["improvements"]["throughput_pct"] == 30.0
    assert comp["improvements"]["memory_pct"] is None
