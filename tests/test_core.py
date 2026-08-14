import pytest
from backend.utils.system import get_system_info
from backend.benchmark.engine import BenchmarkEngine
from backend.inference.engine import InferenceEngine

def test_system_detection():
    info = get_system_info()
    assert "architecture" in info
    assert "cpu_cores_physical" in info
    assert isinstance(info["is_arm"], bool)
    assert info["status_message"] in ["ARM64 ENVIRONMENT \u2014 BENCHMARK ELIGIBLE", "DEVELOPMENT ENVIRONMENT \u2014 x86_64", "UNKNOWN ENVIRONMENT"]

class MockEngine(InferenceEngine):
    def __init__(self):
        self.is_loaded = True

    def load_model(self):
        self.is_loaded = True
        return True

    def unload_model(self):
        self.is_loaded = False

    def generate(self, prompt, max_tokens, temperature):
        # Mock behavior
        return {
            "response": "mock response",
            "latency_ms": 100.0,
            "tokens_generated": 10,
            "tokens_per_second": 100.0,
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
    
    # Run the benchmark
    report = benchmark_engine.run_baseline()
    
    assert "timestamp" in report
    assert "results" in report
    
    res = report["results"]
    assert res["mean_latency_ms"] == 100.0
    assert res["median_latency_ms"] == 100.0
    assert res["p95_latency_ms"] == 100.0
    assert res["mean_tokens_per_second"] == 100.0
    assert len(res["runs"]) == 5
