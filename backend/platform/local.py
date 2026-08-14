import platform
import sys
import psutil
from typing import Dict, Any, Optional
from .base import PlatformAdapter

class LocalPlatformAdapter(PlatformAdapter):
    """Adapter for local host environments (workstations, development laptops, local testbeds)."""

    def __init__(self):
        self._arch = platform.machine().lower()
        self._is_arm = self._arch in ['arm64', 'aarch64']

    def get_provider(self) -> str:
        return "local"

    def get_architecture(self) -> str:
        return self._arch

    def is_arm(self) -> bool:
        return self._is_arm

    def get_cpu_info(self) -> Dict[str, Any]:
        return {
            "model": platform.processor() or "Unknown CPU",
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": sys.version.split('\n')[0]
        }

    def get_memory_info(self) -> Dict[str, Any]:
        return {
            "total_gb": round(psutil.virtual_memory().total / (1024.0 ** 3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024.0 ** 3), 2)
        }

    def get_cloud_provider(self) -> Optional[str]:
        return None

    def get_machine_family(self) -> Optional[str]:
        return None

    def get_machine_type(self) -> Optional[str]:
        return None

    def get_processor_family(self) -> Optional[str]:
        return None

    def get_region_info(self) -> Dict[str, Optional[str]]:
        return {"region": None, "zone": None}
