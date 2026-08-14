import argparse
import time
from backend.optimizer.models import Objective, OptimizationRequest
from backend.optimizer.engine import OptimizationEngine
from backend.api.main import engine, benchmark_engine

def print_table(results):
    print(f"\n{'CONFIG':<10} {'LATENCY (ms)':<15} {'TOK/S':<15} {'MEMORY (MB)':<15} {'SCORE':<10}")
    print("-" * 65)
    for r in results:
        cfg = r['configuration']['threads']
        lat = r['results']['mean_latency_ms']
        tps = r['results']['mean_tokens_per_second']
        mem = r['results'].get('memory_mb', 0)
        score = r.get('score', 0)
        print(f"{cfg:<10} {lat:<15.2f} {tps:<15.2f} {mem:<15.2f} {score:<10.4f}")
    print()

def main():
    parser = argparse.ArgumentParser(description="ARMScale Optimization CLI")
    parser.add_argument("--objective", type=str, required=True, choices=[e.value for e in Objective], help="Optimization objective")
    parser.add_argument("--threads", type=str, help="Comma-separated list of threads to test (e.g., 1,2,4,6,8,12)")
    
    args = parser.parse_request_args() if hasattr(parser, 'parse_request_args') else parser.parse_args()
    
    req = OptimizationRequest(objective=Objective(args.objective))
    if args.threads:
        req.threads_to_test = [int(x.strip()) for x in args.threads.split(",")]
        
    print(f"Initializing optimizer with objective: {req.objective.value}")
    engine.load_model() # Load initial to ensure it works
    
    optimizer = OptimizationEngine(engine, benchmark_engine)
    exp_id = optimizer.start_optimization(req)
    
    print(f"Started experiment {exp_id}. Running...\n")
    
    result = optimizer.run_optimization_sync(req, exp_id)
    
    print("Optimization Complete!\n")
    print_table(result.results)
    
    best = result.best_configuration
    print(f"WINNER ({req.objective.value.upper()}): {best['configuration']['threads']} threads")
    print(f"Improvement vs Baseline Latency:    {result.improvement_vs_baseline['latency_pct']:.2f}%")
    print(f"Improvement vs Baseline Throughput: {result.improvement_vs_baseline['throughput_pct']:.2f}%\n")
    
    print("PARETO CONFIGURATIONS:")
    for p in result.pareto_configurations:
        print(f"- {p['configuration']['threads']} threads (Lat: {p['results']['mean_latency_ms']:.2f}ms, Tok/s: {p['results']['mean_tokens_per_second']:.2f})")
    
if __name__ == "__main__":
    main()
