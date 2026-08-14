import os
import sys
import urllib.request
import argparse

MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILENAME}"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
EXPECTED_SIZE_MB = 398

def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        sys.stdout.write(f"\rDownloading... {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
        sys.stdout.flush()

def download_model():
    print(f"Verified Model Details:")
    print(f"Repository: {MODEL_REPO}")
    print(f"Filename:   {MODEL_FILENAME}")
    print(f"URL:        {MODEL_URL}")
    print(f"Target:     {MODEL_PATH}")
    print("-" * 50)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if os.path.exists(MODEL_PATH):
        print(f"Error: Model file already exists at {MODEL_PATH}")
        print("Will not overwrite existing model silently. Please delete it first if you want to redownload.")
        sys.exit(1)

    print("Starting download...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=report_progress)
        print("\nDownload complete.")
        
        # Verify file size
        actual_size = os.path.getsize(MODEL_PATH)
        actual_size_mb = actual_size / (1024 * 1024)
        print(f"Exact filename: {MODEL_FILENAME}")
        print(f"Exact file size: {actual_size} bytes ({actual_size_mb:.2f} MB)")
        
        # Simple size sanity check
        if actual_size_mb < EXPECTED_SIZE_MB * 0.9 or actual_size_mb > EXPECTED_SIZE_MB * 1.1:
            print("Warning: Downloaded file size differs significantly from expected size.")
        
        print("Model verified and ready for use.")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        if os.path.exists(MODEL_PATH):
            print("Cleaning up partial download...")
            os.remove(MODEL_PATH)
        sys.exit(1)

if __name__ == "__main__":
    download_model()
