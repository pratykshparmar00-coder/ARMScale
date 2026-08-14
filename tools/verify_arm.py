import platform
import sys

def verify_arm():
    arch = platform.machine().lower()
    if arch in ['arm64', 'aarch64']:
        print(f"ARM64/aarch64 -> valid Arm environment (Detected: {arch})")
        return True
    elif arch in ['amd64', 'x86_64']:
        print(f"x86_64/AMD64 -> development environment only (Detected: {arch})")
        return False
    else:
        print(f"Unknown environment -> development environment only (Detected: {arch})")
        return False

if __name__ == "__main__":
    is_arm = verify_arm()
    if not is_arm:
        sys.exit(0) # Not an error, just reporting.
