from typing import List, Optional
from backend.platform.detector import get_platform
from .models import OptimizationConfig, OptimizationDimension
from backend.config import config

VALID_CONTEXT_SIZES = [1024, 2048, 4096]
DEFAULT_OPTIMIZED_THREADS = 4 # Validated optimal baseline thread count

class ConfigurationGenerator:
    def __init__(self):
        self.platform = get_platform()
        cpu_info = self.platform.get_cpu_info()
        self.physical_cores = cpu_info.get("physical_cores") or 4
        self.logical_cores = cpu_info.get("logical_cores") or self.physical_cores

    def generate_thread_candidates(self, override_threads: Optional[List[int]] = None) -> List[int]:
        """Dynamically generates thread counts based on host core capacity."""
        if override_threads:
            return sorted(list(set(override_threads)))
            
        candidates = set([1, 2])
        step = max(1, self.physical_cores // 3)
        for i in range(step, self.physical_cores + 1, step):
            candidates.add(i)
            
        candidates.add(self.physical_cores)
        if self.physical_cores != self.logical_cores:
            candidates.add(self.logical_cores)
            
        return sorted(list(candidates))

    def generate_context_candidates(self, override_contexts: Optional[List[int]] = None) -> List[int]:
        """Generates and validates safe context window sizes."""
        if override_contexts:
            # Validate within safe operational range (512 to 32768)
            return sorted([c for c in override_contexts if 512 <= c <= 32768])
        return sorted(VALID_CONTEXT_SIZES)

    def generate_configurations(
        self, 
        dimension: OptimizationDimension = OptimizationDimension.THREADS,
        override_threads: Optional[List[int]] = None,
        override_contexts: Optional[List[int]] = None,
        fixed_thread_count: int = DEFAULT_OPTIMIZED_THREADS,
        fixed_context_size: int = 2048
    ) -> List[OptimizationConfig]:
        """Generates candidate optimization configurations according to the target dimension."""
        configs = []
        
        if dimension == OptimizationDimension.THREADS:
            thread_candidates = self.generate_thread_candidates(override_threads)
            for t in thread_candidates:
                configs.append(OptimizationConfig(
                    threads=t,
                    context_size=fixed_context_size,
                    batch_size=1
                ))
                
        elif dimension == OptimizationDimension.CONTEXT:
            context_candidates = self.generate_context_candidates(override_contexts)
            for c in context_candidates:
                configs.append(OptimizationConfig(
                    threads=fixed_thread_count,
                    context_size=c,
                    batch_size=1
                ))
                
        elif dimension == OptimizationDimension.COMBINED:
            # Focused combined search space to keep testing fast and deterministic
            threads_pool = override_threads or [2, 4, min(6, self.physical_cores)]
            context_pool = override_contexts or [1024, 2048, 4096]
            for t in sorted(list(set(threads_pool))):
                for c in sorted(list(set(context_pool))):
                    configs.append(OptimizationConfig(
                        threads=t,
                        context_size=c,
                        batch_size=1
                    ))
                    
        return configs
