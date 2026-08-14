from typing import List
from backend.utils.system import get_system_info
from .models import OptimizationConfig
from backend.config import config

class ConfigurationGenerator:
    def __init__(self):
        self.sys_info = get_system_info()
        self.physical_cores = self.sys_info["cpu_cores_physical"]
        self.logical_cores = self.sys_info["cpu_cores_logical"]

    def generate_thread_candidates(self, override_threads: List[int] = None) -> List[int]:
        if override_threads:
            return override_threads
            
        candidates = set([1, 2])
        # Add logical and physical cores as bounds
        
        step = max(1, self.physical_cores // 3)
        for i in range(step, self.physical_cores + 1, step):
            candidates.add(i)
            
        if self.physical_cores != self.logical_cores:
            candidates.add(self.physical_cores)
            candidates.add(self.logical_cores)
            
        return sorted(list(candidates))

    def generate_configurations(self, override_threads: List[int] = None) -> List[OptimizationConfig]:
        thread_candidates = self.generate_thread_candidates(override_threads)
        
        # Only testing thread counts in this phase to keep the workload identical
        configs = []
        for t in thread_candidates:
            configs.append(OptimizationConfig(
                threads=t,
                context_size=config.MODEL_CONTEXT_SIZE,
                batch_size=1
            ))
            
        return configs
