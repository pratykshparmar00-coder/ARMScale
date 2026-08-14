import platform
import os
import sys
import psutil

def get_system_info():
    """Returns detected system information including ARM64 eligibility."""
    arch = platform.machine().lower()
    
    is_arm = arch in ['arm64', 'aarch64']
    is_x86 = arch in ['amd64', 'x86_64']
    
    status_msg = "UNKNOWN ENVIRONMENT"
    if is_arm:
        status_msg = "ARM64 ENVIRONMENT \u2014 BENCHMARK ELIGIBLE"
    elif is_x86:
        status_msg = "DEVELOPMENT ENVIRONMENT \u2014 x86_64"

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
        "status_message": status_msg
    }
