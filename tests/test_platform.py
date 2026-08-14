import pytest
from unittest.mock import patch, MagicMock
from backend.platform.base import PlatformAdapter
from backend.platform.local import LocalPlatformAdapter
from backend.platform.gcp import GCPPlatformAdapter
from backend.platform.detector import PlatformDetector, get_platform

def test_local_platform_adapter():
    adapter = LocalPlatformAdapter()
    assert adapter.get_provider() == "local"
    assert adapter.get_cloud_provider() is None
    assert adapter.get_machine_family() is None
    
    info = adapter.to_dict()
    assert "provider" in info
    assert "architecture" in info
    assert "physical_cores" in info
    assert "ram_gb" in info
    assert info["provider"] == "local"

@patch("urllib.request.urlopen")
def test_gcp_platform_adapter_mocked(mock_urlopen):
    # Mock context manager responses
    def create_mock_response(byte_content):
        cm = MagicMock()
        cm.__enter__.return_value.read.return_value = byte_content
        cm.__enter__.return_value.status = 200
        return cm

    resp_mt = create_mock_response(b"projects/123/zones/us-central1-a/machineTypes/c4a-standard-4")
    resp_zone = create_mock_response(b"projects/123/zones/us-central1-a")
    resp_id = create_mock_response(b"987654321")
    
    mock_urlopen.side_effect = [resp_mt, resp_zone, resp_id]
    
    adapter = GCPPlatformAdapter()
    assert adapter.get_provider() == "gcp"
    assert adapter.get_cloud_provider() == "Google Cloud"
    assert adapter.get_machine_family() == "C4A"
    assert adapter.get_processor_family() == "Google Axion"
    assert adapter.get_machine_type() == "c4a-standard-4"
    
    reg_info = adapter.get_region_info()
    assert reg_info["zone"] == "us-central1-a"
    assert reg_info["region"] == "us-central1"
    
    info = adapter.to_dict()
    assert info["machine_family"] == "C4A"
    assert info["processor_family"] == "Google Axion"

def test_platform_detector_local_fallback():
    with patch.object(PlatformDetector, "is_gcp_metadata_available", return_value=False):
        detected = PlatformDetector.detect()
        assert isinstance(detected, LocalPlatformAdapter)
        assert detected.get_provider() == "local"
