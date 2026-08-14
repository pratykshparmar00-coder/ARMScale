import platform
import os
import sys
import json

try:
    import psutil
    has_psutil = True
except ImportError:
    has_psutil = False

def get_system_info():
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False) if has_psutil else os.cpu_count(),
        "cpu_cores_logical": psutil.cpu_count(logical=True) if has_psutil else os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024.0 ** 3), 2) if has_psutil else "Unknown (psutil not installed)",
        "python_version": sys.version.split('\n')[0],
        "is_arm": platform.machine().lower() in ['arm64', 'aarch64']
    }
    return info

if __name__ == "__main__":
    info = get_system_info()
    print(json.dumps(info, indent=2))
