import pytest
from backend.utils.system import get_system_info
from backend.benchmark.engine import BenchmarkEngine, calculate_percentile
from backend.inference.engine import InferenceEngine

def test_system_detection():
    info = get_system_info()
    assert "architecture" in info
    assert "cpu_cores_physical" in info
    assert isinstance(info["is_arm"], bool)
    assert info["status_message"] in ["ARM64 ENVIRONMENT \u2014 BENCHMARK ELIGIBLE", "DEVELOPMENT ENVIRONMENT \u2014 x86_64", "UNKNOWN ENVIRONMENT"]

def test_calculate_percentile():
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    # Median is 50th percentile = 30.0
    assert calculate_percentile(data, 50.0) == 30.0
    # 95th percentile
    p95 = calculate_percentile(data, 95.0)
    assert p95 == 48.0
    # 0th and 100th
    assert calculate_percentile(data, 0.0) == 10.0
    assert calculate_percentile(data, 100.0) == 50.0

class MockEngine(InferenceEngine):
    def __init__(self):
        self.is_loaded = True
        self.call_count = 0

    def load_model(self):
        self.is_loaded = True
        return True

    def unload_model(self):
        self.is_loaded = False

    def generate(self, prompt, max_tokens, temperature):
        self.call_count += 1
        # Return varying latencies for testing statistics
        lat = 100.0 + (self.call_count * 10.0)
        return {
            "response": "mock response",
            "latency_ms": lat,
            "tokens_generated": 10,
            "tokens_per_second": 10.0 / (lat / 1000.0),
            "model": "mock_model.gguf",
            "runtime": "mock",
            "architecture": "mock_arch",
            "local": True
        }
        
    def benchmark(self):
        pass
        
    def get_model_info(self):
        return {
            "model_name": "mock",
            "repository": "mock/mock",
            "filename": "mock.gguf",
            "quantization": "Q4_K_M",
            "model_size_mb": 100,
            "runtime": "mock",
            "loaded_status": True
        }

def test_benchmark_calculations():
    engine = MockEngine()
    benchmark_engine = BenchmarkEngine(engine)
    
    report = benchmark_engine.run_baseline(save=False)
    
    assert "timestamp" in report
    assert "results" in report
    
    res = report["results"]
    assert "mean_latency_ms" in res
    assert "median_latency_ms" in res
    assert "p95_latency_ms" in res
    assert "min_latency_ms" in res
    assert "max_latency_ms" in res
    assert "std_latency_ms" in res
    assert res["min_latency_ms"] <= res["median_latency_ms"] <= res["max_latency_ms"]
    assert res["memory_mb"] is None
    assert res["memory_status"] == "unavailable"
    assert len(res["runs"]) == 5
