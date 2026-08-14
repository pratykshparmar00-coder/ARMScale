from backend.benchmark.engine import BenchmarkEngine
from backend.inference.llama_cpp_engine import LlamaCppEngine
from backend.api.main import engine as api_engine

def main():
    print("Initializing engine...")
    if api_engine.load_model():
        print("Model loaded successfully.")
        benchmark = BenchmarkEngine(api_engine)
        benchmark.run_baseline()
    else:
        print("Failed to load model. Benchmark aborted.")

if __name__ == "__main__":
    main()
