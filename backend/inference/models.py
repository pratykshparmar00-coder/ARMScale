from pydantic import BaseModel
from typing import Optional, Dict, Any

class ModelVariant(BaseModel):
    name: str
    repository: str
    filename: str
    quantization: str
    expected_size_mb: float
    license: str = "Apache-2.0"
    is_downloaded: bool = False

# Catalog of known model variants (Q4_K_M is the current active baseline)
ACTIVE_MODEL_VARIANT = ModelVariant(
    name="Qwen2.5-0.5B-Instruct",
    repository="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
    quantization="Q4_K_M",
    expected_size_mb=468.64,
    license="Apache-2.0",
    is_downloaded=True
)

AVAILABLE_VARIANTS = {
    "Q4_K_M": ACTIVE_MODEL_VARIANT,
    "Q5_K_M": ModelVariant(
        name="Qwen2.5-0.5B-Instruct",
        repository="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q5_k_m.gguf",
        quantization="Q5_K_M",
        expected_size_mb=520.00,
        license="Apache-2.0",
        is_downloaded=False
    ),
    "Q8_0": ModelVariant(
        name="Qwen2.5-0.5B-Instruct",
        repository="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q8_0.gguf",
        quantization="Q8_0",
        expected_size_mb=700.00,
        license="Apache-2.0",
        is_downloaded=False
    )
}
