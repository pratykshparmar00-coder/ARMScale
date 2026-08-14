from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
import time

from backend.inference.llama_cpp_engine import LlamaCppEngine
from backend.platform.detector import get_platform
from backend.utils.system import get_system_info
from backend.config import config
from backend.benchmark.engine import BenchmarkEngine
from backend.optimizer.engine import OptimizationEngine
from backend.optimizer.models import OptimizationRequest, Objective, OptimizationDimension
from backend.optimizer.config_generator import ConfigurationGenerator
from backend.optimizer.recommender import RecommendationEngine
from backend.optimizer.registry import ExperimentRegistry

app = FastAPI(title="ARMScale API", description="Autonomous Arm64 AI Inference Optimizer")

# Global engine and registry instances
engine = LlamaCppEngine()
benchmark_engine = BenchmarkEngine(engine)
optimizer = OptimizationEngine(engine, benchmark_engine)
registry = ExperimentRegistry()
recommender = RecommendationEngine(registry)
config_gen = ConfigurationGenerator()

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend")

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
        "cpu_cores": sys_info["physical_cores"],
        "ram_gb": sys_info["ram_gb"],
        "arm64": sys_info["is_arm"],
        "provider": sys_info["provider"],
        "inference_available": engine.is_loaded
    }

@app.get("/api/platform")
async def get_platform_endpoint():
    return get_platform().to_dict()

@app.get("/api/system")
async def get_system():
    return get_system_info()

@app.get("/api/optimization/search-space")
async def get_search_space():
    meta = config_gen.get_search_space_metadata()
    return {
        "dimension": "global",
        "quantizations": meta["quantizations"],
        "threads": meta["threads"],
        "contexts": meta["contexts"],
        "total_configurations": meta["total_configurations"]
    }

@app.get("/api/benchmark/latest")
async def get_latest_benchmark(workload_type: str = "short_generation"):
    baseline = optimizer.get_reference_baseline(workload_type=workload_type)
    if not baseline:
        raise HTTPException(status_code=404, detail=f"No recorded baseline benchmark found for workload '{workload_type}'")
    return baseline

@app.get("/api/optimization/latest")
async def get_latest_optimization_endpoint(workload_type: Optional[str] = None, dimension: Optional[str] = None):
    latest = registry.get_latest_experiment(workload_type=workload_type, dimension=dimension)
    if not latest:
        raise HTTPException(status_code=404, detail="No completed optimization experiments found")
    return latest

@app.get("/api/optimization/experiments")
async def list_experiments_endpoint(workload_type: Optional[str] = None):
    return registry.list_experiments(workload_type=workload_type)

@app.get("/api/optimization/pareto")
async def get_pareto_endpoint(workload_type: str = "short_generation", dimension: Optional[str] = None):
    latest = registry.get_latest_experiment(workload_type=workload_type, dimension=dimension)
    if not latest or "pareto_configurations" not in latest:
        raise HTTPException(status_code=404, detail=f"No Pareto configurations found for workload '{workload_type}'")
    return {
        "workload": workload_type,
        "dimension": latest.get("dimension", "unknown"),
        "experiment_id": latest.get("experiment_id"),
        "pareto_configurations": latest.get("pareto_configurations", [])
    }

class RecommendRequest(BaseModel):
    workload: str = "short_generation"
    objective: str = "speed"
    constraints: Optional[Dict[str, Any]] = None

@app.post("/api/optimize/recommend")
async def recommend_endpoint(req: RecommendRequest):
    res = recommender.recommend(workload=req.workload, objective=req.objective, constraints=req.constraints)
    return res

@app.get("/api/model")
async def get_model():
    return engine.get_model_info()

@app.get("/api/models/variants")
async def get_model_variants():
    from backend.inference.models import AVAILABLE_VARIANTS, get_variant_identity
    return {k: get_variant_identity(k) for k in AVAILABLE_VARIANTS.keys()}

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

# Frontend Static Serving
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def serve_ui():
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"status": "ARMScale API running. Frontend index.html not yet initialized."}
