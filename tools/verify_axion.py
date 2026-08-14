import os
import sys
import platform

def verify_axion():
    print("========================================")
    print("ARMScale — Google Axion Arm64 Validator")
    print("========================================\n")
    
    # 1. Check OS
    os_name = platform.system().lower()
    print(f"1. Checking Operating System: {platform.system()} ({platform.release()})")
    if os_name != "linux":
        print(f"   [FAIL] Operating system is {platform.system()}, but Linux is required for Google Axion C4A.")
        print("\nAXION VALIDATION FAILED: OS is not Linux")
        return False
    print("   [PASS] Linux OS confirmed.")

    # 2. Check Architecture
    arch = platform.machine().lower()
    print(f"\n2. Checking Architecture: {arch}")
    if arch not in ["aarch64", "arm64"]:
        print(f"   [FAIL] Detected architecture is '{arch}'. Native aarch64/arm64 is required.")
        print(f"\nAXION VALIDATION FAILED: Architecture is {arch}, expected aarch64/arm64")
        return False
    print("   [PASS] Native Arm64/aarch64 architecture confirmed.")

    # 3. Check llama_cpp installation
    print("\n3. Checking llama-cpp-python native runtime...")
    try:
        from llama_cpp import Llama
        print("   [PASS] llama_cpp imported successfully.")
    except ImportError as e:
        print(f"   [FAIL] Failed to import llama_cpp: {e}")
        print("\nAXION VALIDATION FAILED: llama-cpp-python is not installed or failed to import")
        return False

    # 4. Check Model File
    from backend.config import config
    print(f"\n4. Checking Model File at: {config.MODEL_PATH}")
    if not os.path.exists(config.MODEL_PATH):
        print(f"   [FAIL] Model file not found at {config.MODEL_PATH}")
        print("\nAXION VALIDATION FAILED: Model file missing (run python tools/download_model.py)")
        return False
    model_size_mb = os.path.getsize(config.MODEL_PATH) / (1024 * 1024)
    print(f"   [PASS] Model file exists ({model_size_mb:.2f} MB).")

    # 5. Check Model Loading
    print("\n5. Checking Model Loading...")
    from backend.inference.llama_cpp_engine import LlamaCppEngine
    engine = LlamaCppEngine()
    success = engine.load_model()
    if not success or not engine.is_loaded:
        print("   [FAIL] Model failed to initialize in llama.cpp engine.")
        print("\nAXION VALIDATION FAILED: Model initialization failed")
        return False
    print("   [PASS] Model initialized and resident in memory.")

    # 6. Check Test Generation
    print("\n6. Checking Test Generation...")
    try:
        res = engine.generate("Hello Axion!", max_tokens=16, temperature=0.0)
        print(f"   [PASS] Generation output: {res['response'][:50]}...")
        print(f"   [PASS] Latency: {res['latency_ms']:.2f} ms | Throughput: {res['tokens_per_second']:.2f} tok/s")
    except Exception as e:
        print(f"   [FAIL] Test generation raised exception: {e}")
        print("\nAXION VALIDATION FAILED: Test generation failed")
        return False

    print("\n========================================")
    print("AXION VALIDATION PASSED")
    print("========================================")
    return True

if __name__ == "__main__":
    passed = verify_axion()
    sys.exit(0 if passed else 1)
