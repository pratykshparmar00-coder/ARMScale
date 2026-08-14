from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import time

from backend.inference.llama_cpp_engine import LlamaCppEngine
from backend.utils.system import get_system_info
from backend.config import config
from backend.benchmark.engine import BenchmarkEngine
from backend.optimizer.engine import OptimizationEngine
from backend.optimizer.models import OptimizationRequest, Objective

app = FastAPI(title="ARMScale API", description="Autonomous Arm64 AI Inference Optimizer")

# Global engine instances
engine = LlamaCppEngine()
benchmark_engine = BenchmarkEngine(engine)
optimizer = OptimizationEngine(engine, benchmark_engine)

@app.on_event("startup")
async def startup_event():
    print("Starting ARMScale API...")
    if os.path.exists(config.MODEL_PATH):
        print(f"Loading model from {config.MODEL_PATH}...")
        success = engine.load_model()
        if success:
            print("Model loaded successfully.")
        else:
            print("Warning: Model failed to load.")
    else:
        print(f"Warning: Model file not found at {config.MODEL_PATH}. API starting without resident model.")

@app.on_event("shutdown")
async def shutdown_event():
    engine.unload_model()

@app.get("/health")
async def health_check():
    sys_info = get_system_info()
    return {
        "status": "ok",
        "architecture": sys_info["architecture"],
        "cpu": sys_info["cpu"],
        "cpu_cores": sys_info["cpu_cores_physical"],
        "ram_gb": sys_info["ram_gb"],
        "arm64": sys_info["is_arm"],
        "inference_available": engine.is_loaded
    }

@app.get("/api/system")
async def get_system():
    return get_system_info()

@app.get("/api/benchmark/latest")
async def get_latest_benchmark():
    baseline = optimizer.get_reference_baseline()
    if not baseline:
        raise HTTPException(status_code=404, detail="No recorded baseline benchmark found")
    return baseline

@app.get("/api/model")
async def get_model():
    return engine.get_model_info()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = config.MAX_TOKENS
    temperature: float = config.TEMPERATURE

@app.post("/api/generate")
async def generate_text(req: GenerateRequest):
    if not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is currently unavailable."
        )
        
    try:
        result = engine.generate(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize")
async def start_optimize(req: OptimizationRequest, background_tasks: BackgroundTasks):
    experiment_id = optimizer.start_optimization(req)
    # Run synchronously for now to ensure it completes before user queries, or background it.
    # The instructions say: "Do not block the API request indefinitely. Implement a simple experiment/job system."
    background_tasks.add_task(optimizer.run_optimization_sync, req, experiment_id)
    return {
        "experiment_id": experiment_id,
        "status": "queued"
    }

@app.get("/api/optimize/{experiment_id}")
async def get_optimize_status(experiment_id: str):
    if experiment_id not in optimizer.jobs:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return optimizer.jobs[experiment_id]

