from backend.platform.detector import get_platform

def get_system_info():
    """Returns detected system information from the active PlatformAdapter."""
    return get_platform().to_dict()
