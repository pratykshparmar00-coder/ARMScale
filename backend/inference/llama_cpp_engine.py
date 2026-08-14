import os
import time
from typing import Dict, Any, Optional
from .engine import InferenceEngine
from backend.config import config
from backend.platform.detector import get_platform

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

class LlamaCppEngine(InferenceEngine):
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.model_path = config.MODEL_PATH
        self.active_threads = config.MODEL_THREADS
        self.active_context_size = config.MODEL_CONTEXT_SIZE
        
    def load_model(self, threads: Optional[int] = None, context_size: Optional[int] = None) -> bool:
        if not HAS_LLAMA_CPP:
            print("Error: llama_cpp is not installed.")
            return False
            
        if not os.path.exists(self.model_path):
            print(f"Error: Model not found at {self.model_path}")
            return False
            
        self.active_threads = threads if threads is not None else config.MODEL_THREADS
        self.active_context_size = context_size if context_size is not None else config.MODEL_CONTEXT_SIZE
        
        try:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.active_context_size,
                n_threads=self.active_threads,
                verbose=False
            )
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.is_loaded = False
            return False

    def unload_model(self) -> None:
        if self.model:
            del self.model
            self.model = None
        self.is_loaded = False

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        if not self.is_loaded or not self.model:
            raise RuntimeError("Model is not loaded.")
            
        # Format as ChatML prompt
        formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        start_time = time.perf_counter()
        
        output = self.model(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>"]
        )
        
        end_time = time.perf_counter()
        latency_s = end_time - start_time
        latency_ms = latency_s * 1000.0
        
        text_response = output['choices'][0]['text']
        tokens_generated = output['usage']['completion_tokens']
        
        tokens_per_second = tokens_generated / latency_s if latency_s > 0 else 0
        platform_info = get_platform().to_dict()
        
        return {
            "response": text_response.strip(),
            "latency_ms": latency_ms,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_per_second,
            "model": os.path.basename(self.model_path),
            "runtime": "llama.cpp",
            "architecture": platform_info["architecture"],
            "platform": platform_info["provider"],
            "local": platform_info["provider"] == "local"
        }

    def benchmark(self) -> Dict[str, Any]:
        pass

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "Qwen2.5-0.5B-Instruct",
            "repository": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "filename": os.path.basename(self.model_path),
            "quantization": "Q4_K_M",
            "model_size_mb": os.path.getsize(self.model_path) / (1024*1024) if os.path.exists(self.model_path) else 0,
            "runtime": "llama.cpp",
            "active_threads": self.active_threads,
            "active_context_size": self.active_context_size,
            "loaded_status": self.is_loaded
        }
