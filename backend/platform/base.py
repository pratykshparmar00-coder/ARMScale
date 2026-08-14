from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class PlatformAdapter(ABC):
    """Abstract base class defining the hardware and cloud environment interface."""

    @abstractmethod
    def get_provider(self) -> str:
        """Returns provider identifier: 'local', 'gcp', 'aws', 'generic_arm', etc."""
        pass

    @abstractmethod
    def get_architecture(self) -> str:
        """Returns hardware architecture: 'x86_64', 'amd64', 'arm64', 'aarch64', etc."""
        pass

    @abstractmethod
    def is_arm(self) -> bool:
        """Returns True if running natively on Arm64/aarch64 architecture."""
        pass

    @abstractmethod
    def get_cpu_info(self) -> Dict[str, Any]:
        """Returns CPU model, physical core count, and logical core count."""
        pass

    @abstractmethod
    def get_memory_info(self) -> Dict[str, Any]:
        """Returns total RAM in GB."""
        pass

    @abstractmethod
    def get_cloud_provider(self) -> Optional[str]:
        """Returns human-readable cloud provider name or None if local."""
        pass

    @abstractmethod
    def get_machine_family(self) -> Optional[str]:
        """Returns machine family (e.g., 'C4A') or None."""
        pass

    @abstractmethod
    def get_machine_type(self) -> Optional[str]:
        """Returns machine instance type (e.g., 'c4a-standard-4') or None."""
        pass

    @abstractmethod
    def get_processor_family(self) -> Optional[str]:
        """Returns processor family branding (e.g., 'Google Axion') or None."""
        pass

    @abstractmethod
    def get_region_info(self) -> Dict[str, Optional[str]]:
        """Returns region and zone if running in cloud, or None."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the platform information into a standardized schema."""
        cpu = self.get_cpu_info()
        mem = self.get_memory_info()
        reg = self.get_region_info()
        arch = self.get_architecture()
        is_arm = self.is_arm()
        
        status_msg = "UNKNOWN ENVIRONMENT"
        if is_arm:
            status_msg = "ARM64 ENVIRONMENT — BENCHMARK ELIGIBLE"
        elif arch.lower() in ["amd64", "x86_64"]:
            status_msg = "DEVELOPMENT ENVIRONMENT — x86_64"
            
        return {
            "provider": self.get_provider(),
            "cloud_provider": self.get_cloud_provider(),
            "architecture": arch,
            "is_arm": is_arm,
            "status_message": status_msg,
            "machine_family": self.get_machine_family(),
            "machine_type": self.get_machine_type(),
            "processor_family": self.get_processor_family(),
            "cpu": cpu.get("model", "Unknown"),
            "physical_cores": cpu.get("physical_cores"),
            "logical_cores": cpu.get("logical_cores"),
            "cpu_cores_physical": cpu.get("physical_cores"),
            "cpu_cores_logical": cpu.get("logical_cores"),
            "ram_gb": mem.get("total_gb"),
            "zone": reg.get("zone"),
            "region": reg.get("region"),
            "os": cpu.get("os", "Unknown"),
            "os_release": cpu.get("os_release", "Unknown"),
            "python_version": cpu.get("python_version", "Unknown")
        }
