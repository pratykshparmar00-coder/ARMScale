import os
import hashlib
from pydantic import BaseModel
from typing import Optional, Dict, Any

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")

class ModelVariant(BaseModel):
    name: str
    repository: str
    filename: str
    quantization: str
    expected_bytes: int
    expected_size_mb: float
    license: str = "Apache-2.0"
    quality_score: Optional[float] = None # Placeholder for future quality evaluation benchmark

_SHA256_CACHE: Dict[str, str] = {}

def calculate_file_sha256(filepath: str) -> Optional[str]:
    """Calculates SHA256 checksum of a local file in 64KB chunks with caching."""
    if not os.path.exists(filepath):
        return None
    if filepath in _SHA256_CACHE:
        return _SHA256_CACHE[filepath]
    
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    _SHA256_CACHE[filepath] = digest
    return digest

# Official Catalog of Qwen2.5-0.5B-Instruct-GGUF Variants
AVAILABLE_VARIANTS: Dict[str, ModelVariant] = {
    "Q4_K_M": ModelVariant(
        name="Qwen2.5-0.5B-Instruct",
        repository="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        quantization="Q4_K_M",
        expected_bytes=491400032,
        expected_size_mb=468.64,
        license="Apache-2.0",
        quality_score=None
    ),
    "Q5_K_M": ModelVariant(
        name="Qwen2.5-0.5B-Instruct",
        repository="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q5_k_m.gguf",
        quantization="Q5_K_M",
        expected_bytes=522186592,
        expected_size_mb=497.99,
        license="Apache-2.0",
        quality_score=None
    ),
    "Q8_0": ModelVariant(
        name="Qwen2.5-0.5B-Instruct",
        repository="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q8_0.gguf",
        quantization="Q8_0",
        expected_bytes=675710816,
        expected_size_mb=644.41,
        license="Apache-2.0",
        quality_score=None
    )
}

def get_model_path_for_variant(quantization: str) -> str:
    """Returns the absolute path to the model file for a given quantization format."""
    variant = AVAILABLE_VARIANTS.get(quantization.upper())
    if not variant:
        raise ValueError(f"Unknown quantization variant: '{quantization}'. Available: {list(AVAILABLE_VARIANTS.keys())}")
    return os.path.join(MODEL_DIR, variant.filename)

def is_variant_downloaded(quantization: str) -> bool:
    """Checks if the exact model file exists locally and is non-empty."""
    path = get_model_path_for_variant(quantization)
    return os.path.exists(path) and os.path.getsize(path) > 0

def get_variant_identity(quantization: str) -> Dict[str, Any]:
    """Returns complete identity and verification metadata for a model variant."""
    variant = AVAILABLE_VARIANTS.get(quantization.upper())
    if not variant:
        return {"quantization": quantization, "status": "unknown"}
    
    path = get_model_path_for_variant(quantization)
    exists = os.path.exists(path)
    file_bytes = os.path.getsize(path) if exists else 0
    size_mb = file_bytes / (1024 * 1024) if exists else 0.0
    sha256_hash = calculate_file_sha256(path) if exists else None
    
    return {
        "repository": variant.repository,
        "filename": variant.filename,
        "quantization": variant.quantization,
        "filepath": path,
        "file_size_bytes": file_bytes,
        "model_size_mb": round(size_mb, 2),
        "sha256": sha256_hash,
        "license": variant.license,
        "quality_score": variant.quality_score,
        "is_downloaded": exists and file_bytes > 0
    }
