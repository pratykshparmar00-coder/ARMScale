from typing import List, Optional
from backend.platform.detector import get_platform
from backend.inference.models import AVAILABLE_VARIANTS
from .models import OptimizationConfig, OptimizationDimension
from backend.config import config

VALID_THREADS_MATRIX = [2, 4, 6, 8]
VALID_CONTEXT_MATRIX = [1024, 2048, 4096]
VALID_QUANTIZATION_MATRIX = ["Q4_K_M", "Q5_K_M", "Q8_0"]
DEFAULT_OPTIMIZED_THREADS = 6 # Optimal thread count discovered in Phase F
DEFAULT_OPTIMIZED_CONTEXT = 4096 # Optimal context window discovered in Phase F

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
            return sorted([c for c in override_contexts if 512 <= c <= 32768])
        return sorted(VALID_CONTEXT_MATRIX)

    def generate_quantization_candidates(self, override_quantizations: Optional[List[str]] = None) -> List[str]:
        """Validates and returns quantization candidates from the verified catalog."""
        if override_quantizations:
            valid = []
            for q in override_quantizations:
                q_upper = q.upper()
                if q_upper in AVAILABLE_VARIANTS:
                    valid.append(q_upper)
            return valid if valid else ["Q4_K_M"]
        return list(VALID_QUANTIZATION_MATRIX)

    def generate_configurations(
        self, 
        dimension: OptimizationDimension = OptimizationDimension.THREADS,
        override_threads: Optional[List[int]] = None,
        override_contexts: Optional[List[int]] = None,
        override_quantizations: Optional[List[str]] = None,
        fixed_thread_count: int = DEFAULT_OPTIMIZED_THREADS,
        fixed_context_size: int = DEFAULT_OPTIMIZED_CONTEXT,
        fixed_quantization: str = "Q4_K_M"
    ) -> List[OptimizationConfig]:
        """Generates candidate optimization configurations according to the target dimension."""
        configs = []
        
        if dimension == OptimizationDimension.THREADS:
            thread_candidates = self.generate_thread_candidates(override_threads)
            for t in thread_candidates:
                configs.append(OptimizationConfig(
                    threads=t,
                    context_size=fixed_context_size,
                    quantization=fixed_quantization,
                    batch_size=1
                ))
                
        elif dimension == OptimizationDimension.CONTEXT:
            context_candidates = self.generate_context_candidates(override_contexts)
            for c in context_candidates:
                configs.append(OptimizationConfig(
                    threads=fixed_thread_count,
                    context_size=c,
                    quantization=fixed_quantization,
                    batch_size=1
                ))

        elif dimension == OptimizationDimension.QUANTIZATION:
            quant_candidates = self.generate_quantization_candidates(override_quantizations)
            for q in quant_candidates:
                configs.append(OptimizationConfig(
                    threads=fixed_thread_count,
                    context_size=fixed_context_size,
                    quantization=q,
                    batch_size=1
                ))
                
        elif dimension == OptimizationDimension.COMBINED:
            # Multi-dimensional search across threads and context, or threads x context x quantization
            threads_pool = override_threads if override_threads is not None else VALID_THREADS_MATRIX
            context_pool = override_contexts if override_contexts is not None else VALID_CONTEXT_MATRIX
            quant_pool = override_quantizations if override_quantizations is not None else [fixed_quantization]
            
            for q in quant_pool:
                for t in sorted(list(set(threads_pool))):
                    for c in sorted(list(set(context_pool))):
                        configs.append(OptimizationConfig(
                            threads=t,
                            context_size=c,
                            quantization=q,
                            batch_size=1
                        ))
                    
        return configs
