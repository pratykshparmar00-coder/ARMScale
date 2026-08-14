from pydantic import BaseModel, validator
from typing import Dict, Any, List, Optional
from enum import Enum

class Objective(str, Enum):
    SPEED = "speed"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    BALANCED = "balanced"
    SIZE = "size"

class OptimizationDimension(str, Enum):
    THREADS = "threads"
    CONTEXT = "context"
    QUANTIZATION = "quantization"
    COMBINED = "combined"
    GLOBAL = "global"

VALID_QUANTIZATIONS = ["Q4_K_M", "Q5_K_M", "Q8_0"]

class OptimizationConfig(BaseModel):
    quantization: str = "Q4_K_M"
    threads: int
    context_size: int
    batch_size: int = 1
    configuration_id: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        if not self.configuration_id:
            self.configuration_id = f"cfg_{self.quantization.upper()}_T{self.threads}_C{self.context_size}"

    @validator('quantization')
    def validate_quantization(cls, v):
        v_upper = v.upper()
        if v_upper not in VALID_QUANTIZATIONS:
            raise ValueError(f"Invalid quantization '{v}'. Supported: {VALID_QUANTIZATIONS}")
        return v_upper

    @validator('threads')
    def validate_threads(cls, v):
        if v <= 0:
            raise ValueError(f"Threads must be a positive integer, got {v}")
        return v

    @validator('context_size')
    def validate_context_size(cls, v):
        if v <= 0:
            raise ValueError(f"Context size must be a positive integer, got {v}")
        return v

    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v != 1:
            raise ValueError(f"Batch size must be 1 for current serving engine, got {v}")
        return v

    def dict(self, **kwargs):
        return {
            "configuration_id": self.configuration_id or f"cfg_{self.quantization.upper()}_T{self.threads}_C{self.context_size}",
            "quantization": self.quantization,
            "threads": self.threads,
            "context_size": self.context_size,
            "batch_size": self.batch_size
        }

class OptimizationRequest(BaseModel):
    objective: Objective = Objective.SPEED
    dimension: OptimizationDimension = OptimizationDimension.THREADS
    workload_type: str = "short_generation"
    max_memory_mb: Optional[int] = None
    threads_to_test: Optional[List[int]] = None
    context_sizes_to_test: Optional[List[int]] = None
    quantizations_to_test: Optional[List[str]] = None

    @validator('workload_type')
    def validate_workload(cls, v):
        if v not in ["short_generation", "context_stress"]:
            raise ValueError(f"Invalid workload '{v}'. Supported: 'short_generation', 'context_stress'")
        return v

class OptimizationResult(BaseModel):
    experiment_id: str
    dimension: str
    workload_type: str
    platform: Dict[str, Any]
    search_space: Optional[Dict[str, Any]] = None
    baseline: Dict[str, Any]
    configurations_tested: int
    results: List[Dict[str, Any]]
    best_configuration: Dict[str, Any]
    pareto_configurations: List[Dict[str, Any]]
    improvement_vs_baseline: Dict[str, Optional[float]]
    execution_time_s: float
