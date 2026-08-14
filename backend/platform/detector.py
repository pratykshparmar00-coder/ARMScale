import urllib.request
import urllib.error
from .base import PlatformAdapter
from .local import LocalPlatformAdapter
from .gcp import GCPPlatformAdapter

class PlatformDetector:
    """Detects the execution environment and returns the appropriate PlatformAdapter."""

    @staticmethod
    def is_gcp_metadata_available() -> bool:
        """Checks if the GCP Compute Engine metadata server is reachable."""
        url = "http://metadata.google.internal/computeMetadata/v1/instance/id"
        headers = {"Metadata-Flavor": "Google"}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=0.2) as resp:
                return resp.status == 200
        except Exception:
            return False

    @classmethod
    def detect(cls) -> PlatformAdapter:
        """Returns GCPPlatformAdapter if inside GCP, else LocalPlatformAdapter."""
        if cls.is_gcp_metadata_available():
            return GCPPlatformAdapter()
        return LocalPlatformAdapter()

# Singleton accessor
_current_platform = None

def get_platform() -> PlatformAdapter:
    global _current_platform
    if _current_platform is None:
        _current_platform = PlatformDetector.detect()
    return _current_platform
