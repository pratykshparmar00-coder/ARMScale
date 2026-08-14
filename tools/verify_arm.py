import platform
import sys

def verify_arm():
    print("========================================")
    print("ARMScale — Generic Arm64 Validator")
    print("========================================\n")
    
    arch = platform.machine().lower()
    print(f"Detected Architecture: {arch}")
    
    if arch in ['arm64', 'aarch64']:
        print("Architecture Check: Native Arm64/aarch64 confirmed.")
        print("\n========================================")
        print("ARM64 VALIDATION PASSED")
        print("========================================")
        return True
    else:
        print(f"Architecture Check: '{arch}' is not an Arm64 architecture.")
        print("\n========================================")
        print("ARM64 VALIDATION FAILED")
        print("========================================")
        return False

if __name__ == "__main__":
    passed = verify_arm()
    sys.exit(0 if passed else 1)
