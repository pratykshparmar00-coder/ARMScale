import platform
import sys
import psutil
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
from .base import PlatformAdapter

class GCPPlatformAdapter(PlatformAdapter):
    """Adapter for Google Cloud Compute Engine instances (e.g., C4A Google Axion)."""

    def __init__(self):
        self._arch = platform.machine().lower()
        self._is_arm = self._arch in ['arm64', 'aarch64']
        self._metadata = self._fetch_gcp_metadata()

    def _fetch_gcp_metadata(self) -> Dict[str, Any]:
        base_url = "http://metadata.google.internal/computeMetadata/v1/instance/"
        headers = {"Metadata-Flavor": "Google"}
        
        meta = {
            "machine_family": None,
            "machine_type": None,
            "processor_family": None,
            "zone": None,
            "region": None,
            "instance_id": None
        }
        
        try:
            # Machine type
            req = urllib.request.Request(base_url + "machine-type", headers=headers)
            with urllib.request.urlopen(req, timeout=0.2) as resp:
                mt_full = resp.read().decode('utf-8').strip()
                mt = mt_full.split('/')[-1]
                meta["machine_type"] = mt
                if mt.startswith("c4a"):
                    meta["machine_family"] = "C4A"
                    meta["processor_family"] = "Google Axion"
                elif "-" in mt:
                    meta["machine_family"] = mt.split('-')[0].upper()
                    
            # Zone & Region
            req_zone = urllib.request.Request(base_url + "zone", headers=headers)
            with urllib.request.urlopen(req_zone, timeout=0.2) as resp:
                z_full = resp.read().decode('utf-8').strip()
                zone = z_full.split('/')[-1]
                meta["zone"] = zone
                if '-' in zone:
                    meta["region"] = '-'.join(zone.split('-')[:-1])
                    
            # Instance ID
            req_id = urllib.request.Request(base_url + "id", headers=headers)
            with urllib.request.urlopen(req_id, timeout=0.2) as resp:
                meta["instance_id"] = resp.read().decode('utf-8').strip()
                
        except Exception:
            pass
            
        return meta

    def get_provider(self) -> str:
        return "gcp"

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
        return "Google Cloud"

    def get_machine_family(self) -> Optional[str]:
        return self._metadata.get("machine_family")

    def get_machine_type(self) -> Optional[str]:
        return self._metadata.get("machine_type")

    def get_processor_family(self) -> Optional[str]:
        return self._metadata.get("processor_family")

    def get_region_info(self) -> Dict[str, Optional[str]]:
        return {
            "region": self._metadata.get("region"),
            "zone": self._metadata.get("zone")
        }
