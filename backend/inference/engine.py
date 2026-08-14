from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class InferenceEngine(ABC):
    """
    Abstract base class for inference engines.
    The rest of the application must not depend directly on specific runtimes.
    """
    
    @abstractmethod
    def load_model(self) -> bool:
        """Loads the model into memory. Returns True if successful."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unloads the model from memory."""
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """
        Generates text based on the prompt.
        Should return a dictionary containing at least:
        - response: str
        - latency_ms: float (measured inference time)
        - tokens_generated: int
        - tokens_per_second: float
        """
        pass

    @abstractmethod
    def benchmark(self) -> Dict[str, Any]:
        """Runs a baseline performance benchmark."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata about the currently loaded model."""
        pass
