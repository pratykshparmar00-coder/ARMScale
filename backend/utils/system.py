import platform
import os
import sys
import psutil
import urllib.request
import urllib.error
import json

def get_gcp_metadata():
    """Queries GCP Metadata server if running inside a Google Compute Engine instance."""
    base_url = "http://metadata.google.internal/computeMetadata/v1/instance/"
    headers = {"Metadata-Flavor": "Google"}
    
    metadata = {
        "cloud_provider": "None",
        "machine_family": "Unknown",
        "machine_type": "Unknown",
        "processor_family": "Unknown",
        "zone": "Unknown"
    }
    
    try:
        req = urllib.request.Request(base_url + "machine-type", headers=headers)
        with urllib.request.urlopen(req, timeout=0.2) as resp:
            machine_type_full = resp.read().decode('utf-8').strip()
            machine_type = machine_type_full.split('/')[-1]
            metadata["cloud_provider"] = "Google Cloud"
            metadata["machine_type"] = machine_type
            if machine_type.startswith("c4a"):
                metadata["machine_family"] = "C4A"
                metadata["processor_family"] = "Google Axion"
            else:
                metadata["machine_family"] = machine_type.split('-')[0].upper()
                
        req_zone = urllib.request.Request(base_url + "zone", headers=headers)
        with urllib.request.urlopen(req_zone, timeout=0.2) as resp:
            zone_full = resp.read().decode('utf-8').strip()
            metadata["zone"] = zone_full.split('/')[-1]
            
    except (urllib.error.URLError, Exception):
        pass # Not running in GCP or metadata server unreachable
        
    return metadata

def get_system_info():
    """Returns detected system information including ARM64 eligibility and cloud metadata."""
    arch = platform.machine().lower()
    
    is_arm = arch in ['arm64', 'aarch64']
    is_x86 = arch in ['amd64', 'x86_64']
    
    status_msg = "UNKNOWN ENVIRONMENT"
    if is_arm:
        status_msg = "ARM64 ENVIRONMENT — BENCHMARK ELIGIBLE"
    elif is_x86:
        status_msg = "DEVELOPMENT ENVIRONMENT — x86_64"

    gcp_info = get_gcp_metadata()

    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": arch,
        "cpu": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024.0 ** 3), 2),
        "python_version": sys.version.split('\n')[0],
        "is_arm": is_arm,
        "status_message": status_msg,
        "cloud_provider": gcp_info["cloud_provider"],
        "machine_family": gcp_info["machine_family"],
        "machine_type": gcp_info["machine_type"],
        "processor_family": gcp_info["processor_family"],
        "zone": gcp_info["zone"]
    }
