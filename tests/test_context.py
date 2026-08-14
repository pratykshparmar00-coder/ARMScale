import pytest
from backend.optimizer.config_generator import ConfigurationGenerator
from backend.optimizer.models import OptimizationDimension, OptimizationConfig
from backend.benchmark.engine import WorkloadType, SHORT_GENERATION_PROMPTS, CONTEXT_STRESS_PROMPTS

def test_context_candidate_generation():
    gen = ConfigurationGenerator()
    candidates = gen.generate_context_candidates()
    assert 1024 in candidates
    assert 2048 in candidates
    assert 4096 in candidates

    # Test override with safe validation
    override = gen.generate_context_candidates([256, 1024, 2048, 65536])
    assert 1024 in override
    assert 2048 in override
    assert 256 not in override # Filtered (below 512)
    assert 65536 not in override # Filtered (above 32768)

def test_context_dimension_configurations():
    gen = ConfigurationGenerator()
    configs = gen.generate_configurations(dimension=OptimizationDimension.CONTEXT, fixed_thread_count=4)
    assert len(configs) >= 3
    for c in configs:
        assert c.threads == 4
        assert c.context_size in [1024, 2048, 4096]

def test_combined_dimension_configurations():
    gen = ConfigurationGenerator()
    configs = gen.generate_configurations(
        dimension=OptimizationDimension.COMBINED,
        override_threads=[2, 4],
        override_contexts=[1024, 2048]
    )
    assert len(configs) == 4
    pairs = [(c.threads, c.context_size) for c in configs]
    assert (2, 1024) in pairs
    assert (2, 2048) in pairs
    assert (4, 1024) in pairs
    assert (4, 2048) in pairs

def test_workload_definitions():
    assert len(SHORT_GENERATION_PROMPTS) == 5
    assert len(CONTEXT_STRESS_PROMPTS) == 5
    # Context stress prompts should have substantially longer input length
    assert all(len(p) > 300 for p in CONTEXT_STRESS_PROMPTS)
    assert all(len(p) < 150 for p in SHORT_GENERATION_PROMPTS)
