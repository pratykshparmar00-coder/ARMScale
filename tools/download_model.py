import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import urllib.request
from backend.inference.models import AVAILABLE_VARIANTS, calculate_file_sha256, MODEL_DIR

def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        sys.stdout.write(f"\rDownloading... {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
        sys.stdout.flush()

def download_single_variant(quantization: str, force: bool = False) -> bool:
    variant_key = quantization.upper()
    if variant_key not in AVAILABLE_VARIANTS:
        print(f"Error: Unknown quantization '{quantization}'. Available: {list(AVAILABLE_VARIANTS.keys())}")
        return False
        
    variant = AVAILABLE_VARIANTS[variant_key]
    model_url = f"https://huggingface.co/{variant.repository}/resolve/main/{variant.filename}"
    target_path = os.path.join(MODEL_DIR, variant.filename)
    
    print(f"\n========================================")
    print(f"Model Variant: {variant.quantization}")
    print(f"Repository:    {variant.repository}")
    print(f"Filename:      {variant.filename}")
    print(f"Expected Size: {variant.expected_size_mb} MB ({variant.expected_bytes} bytes)")
    print(f"URL:           {model_url}")
    print(f"Target Path:   {target_path}")
    print(f"========================================")

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 0 and not force:
        actual_size = os.path.getsize(target_path)
        print(f"File already exists ({actual_size} bytes). Skipping download.")
        print("Calculating SHA256 checksum...")
        sha = calculate_file_sha256(target_path)
        print(f"SHA256: {sha}")
        return True

    print("Starting download from HuggingFace...")
    try:
        urllib.request.urlretrieve(model_url, target_path, reporthook=report_progress)
        print("\nDownload complete.")
        
        actual_size = os.path.getsize(target_path)
        actual_size_mb = actual_size / (1024 * 1024)
        print(f"Exact file size: {actual_size} bytes ({actual_size_mb:.2f} MB)")
        
        # Verify size sanity
        if abs(actual_size - variant.expected_bytes) > (variant.expected_bytes * 0.05):
            print(f"Warning: Downloaded size ({actual_size}) differs from expected ({variant.expected_bytes}).")
            
        print("Calculating SHA256 checksum...")
        sha = calculate_file_sha256(target_path)
        print(f"SHA256: {sha}")
        print(f"Variant {variant.quantization} verified and ready for use.")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        if os.path.exists(target_path):
            print("Cleaning up partial file...")
            os.remove(target_path)
        return False

def main():
    parser = argparse.ArgumentParser(description="Download and verify Qwen2.5 GGUF quantization variants")
    parser.add_argument("--variant", type=str, default="Q4_K_M", choices=list(AVAILABLE_VARIANTS.keys()) + ["all"], help="Quantization format to download")
    parser.add_argument("--force", action="store_true", help="Force redownload even if file exists")
    
    args = parser.parse_args()
    
    if args.variant == "all":
        for v in AVAILABLE_VARIANTS.keys():
            success = download_single_variant(v, force=args.force)
            if not success:
                sys.exit(1)
    else:
        success = download_single_variant(args.variant, force=args.force)
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()
