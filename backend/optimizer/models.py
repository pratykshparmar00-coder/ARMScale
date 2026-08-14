from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from enum import Enum

class Objective(str, Enum):
    SPEED = "speed"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    BALANCED = "balanced"

class OptimizationDimension(str, Enum):
    THREADS = "threads"
    CONTEXT = "context"
    COMBINED = "combined"

class OptimizationConfig(BaseModel):
    threads: int
    context_size: int
    batch_size: int = 1
    
    def dict(self, **kwargs):
        return {
            "threads": self.threads,
            "context_size": self.context_size,
            "batch_size": self.batch_size
        }

class OptimizationRequest(BaseModel):
    objective: Objective = Objective.SPEED
    dimension: OptimizationDimension = OptimizationDimension.THREADS
    workload_type: str = "short_generation" # "short_generation" or "context_stress"
    max_memory_mb: Optional[int] = None
    threads_to_test: Optional[List[int]] = None
    context_sizes_to_test: Optional[List[int]] = None

class OptimizationResult(BaseModel):
    experiment_id: str
    dimension: str
    workload_type: str
    platform: Dict[str, Any]
    baseline: Dict[str, Any]
    configurations_tested: int
    results: List[Dict[str, Any]]
    best_configuration: Dict[str, Any]
    pareto_configurations: List[Dict[str, Any]]
    improvement_vs_baseline: Dict[str, Optional[float]]
    execution_time_s: float
