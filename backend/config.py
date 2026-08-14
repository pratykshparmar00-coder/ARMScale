import os

def get_env_var(key, default):
    return os.environ.get(key, default)

class Config:
    MODEL_PATH = get_env_var("MODEL_PATH", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    MODEL_THREADS = int(get_env_var("MODEL_THREADS", "4"))
    MODEL_CONTEXT_SIZE = int(get_env_var("MODEL_CONTEXT_SIZE", "2048"))
    MAX_TOKENS = int(get_env_var("MAX_TOKENS", "128"))
    TEMPERATURE = float(get_env_var("TEMPERATURE", "0.7"))
    HOST = get_env_var("HOST", "0.0.0.0")
    PORT = int(get_env_var("PORT", "8000"))

config = Config()
