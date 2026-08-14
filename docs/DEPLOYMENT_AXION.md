# Google Axion C4A Arm64 Deployment Guide

This guide details the exact steps to provision a Google Axion Arm64 VM and execute the ARMScale benchmark and optimization suite.

---

## Step 1: Provision Google Axion C4A VM

Using the `gcloud` CLI or Google Cloud Console:

```bash
gcloud compute instances create armscale-axion-benchmark \
    --project="YOUR_PROJECT_ID" \
    --zone="us-central1-a" \
    --machine-type="c4a-standard-4" \
    --image-family="ubuntu-2404-lts-arm64" \
    --image-project="ubuntu-os-cloud" \
    --boot-disk-size="50GB" \
    --boot-disk-type="pd-balanced"
```

---

## Step 2: Connect to the Instance

```bash
gcloud compute ssh armscale-axion-benchmark --zone="us-central1-a"
```

---

## Step 3: Clone Repository & Setup Environment

```bash
# Update system packages & install build essentials
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git build-essential cmake

# Clone the repository
git clone https://github.com/pratykshparmar00-coder/ARMScale.git
cd ARMScale

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and native aarch64 llama-cpp-python
pip install --upgrade pip
pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_NATIVE=ON" pip install llama-cpp-python
```

---

## Step 4: Download Model & Run Validation

```bash
# Download the verified Qwen2.5-0.5B-Instruct-GGUF model
python tools/download_model.py

# Run Axion Arm64 validation checks
python tools/verify_axion.py
```

Expected output: `AXION VALIDATION PASSED`

---

## Step 5: Run Baseline Benchmark

```bash
python run_benchmark.py
```

---

## Step 6: Run Optimizer Sweep

```bash
# Run hardware-aware thread optimization
python tools/optimize.py --objective speed
```

---

## Step 7: Clean Up Cloud Resources

To avoid incurring cloud compute charges after the benchmark is completed:

```bash
# Stop VM
gcloud compute instances stop armscale-axion-benchmark --zone="us-central1-a"

# Or delete VM
gcloud compute instances delete armscale-axion-benchmark --zone="us-central1-a" --quiet
```
