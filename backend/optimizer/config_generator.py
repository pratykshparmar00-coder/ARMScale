from typing import List, Optional, Dict, Any
from backend.platform.detector import get_platform
from backend.inference.models import AVAILABLE_VARIANTS
from .models import OptimizationConfig, OptimizationDimension, VALID_QUANTIZATIONS
from backend.config import config

GLOBAL_THREADS_MATRIX = [2, 4, 6, 8]
GLOBAL_CONTEXT_MATRIX = [1024, 2048, 4096]
GLOBAL_QUANTIZATION_MATRIX = ["Q4_K_M", "Q5_K_M", "Q8_0"]

DEFAULT_OPTIMIZED_THREADS = 6
DEFAULT_OPTIMIZED_CONTEXT = 4096

class ConfigurationGenerator:
    def __init__(self):
        self.platform = get_platform()
        cpu_info = self.platform.get_cpu_info()
        self.physical_cores = cpu_info.get("physical_cores") or 4
        self.logical_cores = cpu_info.get("logical_cores") or self.physical_cores

    def generate_thread_candidates(self, override_threads: Optional[List[int]] = None) -> List[int]:
        """Dynamically generates thread counts based on host core capacity."""
        if override_threads:
            valid = [t for t in set(override_threads) if isinstance(t, int) and t > 0]
            if not valid:
                raise ValueError("No valid positive thread values provided.")
            return sorted(valid)
            
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
            valid = [c for c in set(override_contexts) if isinstance(c, int) and 512 <= c <= 32768]
            if not valid:
                raise ValueError("No valid context sizes provided (must be between 512 and 32768).")
            return sorted(valid)
        return sorted(GLOBAL_CONTEXT_MATRIX)

    def generate_quantization_candidates(self, override_quantizations: Optional[List[str]] = None) -> List[str]:
        """Validates and returns quantization candidates from the verified catalog."""
        if override_quantizations:
            valid = []
            for q in override_quantizations:
                q_upper = q.upper()
                if q_upper in VALID_QUANTIZATIONS and q_upper in AVAILABLE_VARIANTS:
                    valid.append(q_upper)
            if not valid:
                raise ValueError(f"No valid quantization variants in {override_quantizations}. Supported: {VALID_QUANTIZATIONS}")
            return sorted(list(set(valid)), key=lambda x: VALID_QUANTIZATIONS.index(x) if x in VALID_QUANTIZATIONS else 99)
        return list(GLOBAL_QUANTIZATION_MATRIX)

    def get_search_space_metadata(
        self,
        override_quantizations: Optional[List[str]] = None,
        override_threads: Optional[List[int]] = None,
        override_contexts: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Returns the search space definition and total configuration count."""
        quants = self.generate_quantization_candidates(override_quantizations)
        threads = sorted(list(set(override_threads))) if override_threads else GLOBAL_THREADS_MATRIX
        contexts = sorted(list(set(override_contexts))) if override_contexts else GLOBAL_CONTEXT_MATRIX
        return {
            "quantizations": quants,
            "threads": threads,
            "contexts": contexts,
            "total_configurations": len(quants) * len(threads) * len(contexts)
        }

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
                    quantization=fixed_quantization,
                    threads=t,
                    context_size=fixed_context_size,
                    batch_size=1
                ))
                
        elif dimension == OptimizationDimension.CONTEXT:
            context_candidates = self.generate_context_candidates(override_contexts)
            for c in context_candidates:
                configs.append(OptimizationConfig(
                    quantization=fixed_quantization,
                    threads=fixed_thread_count,
                    context_size=c,
                    batch_size=1
                ))

        elif dimension == OptimizationDimension.QUANTIZATION:
            quant_candidates = self.generate_quantization_candidates(override_quantizations)
            for q in quant_candidates:
                configs.append(OptimizationConfig(
                    quantization=q,
                    threads=fixed_thread_count,
                    context_size=fixed_context_size,
                    batch_size=1
                ))
                
        elif dimension == OptimizationDimension.COMBINED:
            # 2-Dimensional Joint Matrix: Threads x Context (with fixed or specified quantization)
            quants = override_quantizations if override_quantizations is not None else [fixed_quantization]
            threads = sorted(list(set(override_threads))) if override_threads else GLOBAL_THREADS_MATRIX
            contexts = sorted(list(set(override_contexts))) if override_contexts else GLOBAL_CONTEXT_MATRIX
            
            for q in quants:
                for t in threads:
                    for c in contexts:
                        configs.append(OptimizationConfig(
                            quantization=q,
                            threads=t,
                            context_size=c,
                            batch_size=1
                        ))

        elif dimension == OptimizationDimension.GLOBAL:
            # 36-Configuration Cartesian Product: Quantizations (3) x Threads (4) x Context (3)
            quants = self.generate_quantization_candidates(override_quantizations)
            threads = sorted(list(set(override_threads))) if override_threads else GLOBAL_THREADS_MATRIX
            contexts = sorted(list(set(override_contexts))) if override_contexts else GLOBAL_CONTEXT_MATRIX
            
            for q in quants:
                for t in threads:
                    for c in contexts:
                        configs.append(OptimizationConfig(
                            quantization=q,
                            threads=t,
                            context_size=c,
                            batch_size=1
                        ))
                    
        return configs
