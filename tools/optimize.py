import argparse
import time
from backend.optimizer.models import Objective, OptimizationRequest
from backend.optimizer.engine import OptimizationEngine
from backend.api.main import engine, benchmark_engine

def print_table(results):
    headers = [
        "CONFIG", "MEAN (ms)", "MEDIAN (ms)", "P95 (ms)", "MIN (ms)", "MAX (ms)", "STD (ms)", "TOK/S", "MEMORY", "SCORE"
    ]
    header_fmt = "{:<8} {:<12} {:<12} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<8}"
    print("\n" + header_fmt.format(*headers))
    print("-" * 106)
    
    for r in results:
        cfg = f"{r['configuration']['threads']} th"
        res = r['results']
        mean_l = f"{res['mean_latency_ms']:.2f}"
        med_l = f"{res['median_latency_ms']:.2f}"
        p95_l = f"{res['p95_latency_ms']:.2f}"
        min_l = f"{res['min_latency_ms']:.2f}"
        max_l = f"{res['max_latency_ms']:.2f}"
        std_l = f"{res['std_latency_ms']:.2f}"
        tps = f"{res['mean_tokens_per_second']:.2f}"
        mem = f"{res['memory_mb']:.2f}" if res.get('memory_mb') is not None else "N/A"
        score = f"{r.get('score', 0):.4f}"
        print(header_fmt.format(cfg, mean_l, med_l, p95_l, min_l, max_l, std_l, tps, mem, score))
    print()

def main():
    parser = argparse.ArgumentParser(description="ARMScale Optimization CLI (x86_64 Dev / Arm64 Target)")
    parser.add_argument("--objective", type=str, required=True, choices=[e.value for e in Objective], help="Optimization objective (speed, throughput, balanced, memory)")
    parser.add_argument("--threads", type=str, help="Comma-separated list of thread counts to test (e.g., 1,2,4,6,8,12)")
    
    args = parser.parse_args()
    
    req = OptimizationRequest(objective=Objective(args.objective))
    if args.threads:
        req.threads_to_test = [int(x.strip()) for x in args.threads.split(",")]
        
    print(f"=== ARMScale Optimization Engine ===")
    print(f"Target Objective: {req.objective.value.upper()}")
    print("Environment: DEVELOPMENT MACHINE — x86_64\n")
    
    engine.load_model()
    
    optimizer = OptimizationEngine(engine, benchmark_engine)
    exp_id = optimizer.start_optimization(req)
    
    print(f"Started experiment {exp_id}. Running benchmark sweep...")
    result = optimizer.run_optimization_sync(req, exp_id)
    
    print("Optimization Sweep Complete!\n")
    print_table(result.results)
    
    best = result.best_configuration
    best_th = best['configuration']['threads']
    print(f"WINNER ({req.objective.value.upper()}): {best_th} threads")
    
    base_res = result.baseline['results']
    print(f"\n--- Baseline Reference (Threads: {result.baseline['configuration']['threads']}) ---")
    print(f"Baseline Mean Latency:    {base_res['mean_latency_ms']:.2f} ms")
    print(f"Baseline Mean Throughput: {base_res['mean_tokens_per_second']:.2f} tok/s")
    
    print(f"\n--- Improvement vs Baseline ---")
    print(f"Latency Improvement:    {result.improvement_vs_baseline['latency_pct']:.2f}%")
    print(f"Throughput Improvement: {result.improvement_vs_baseline['throughput_pct']:.2f}%")
    mem_imp = result.improvement_vs_baseline.get('memory_pct')
    print(f"Memory Improvement:     {f'{mem_imp:.2f}%' if mem_imp is not None else 'N/A (unavailable)'}\n")
    
    print("PARETO CONFIGURATIONS:")
    for p in result.pareto_configurations:
        th = p['configuration']['threads']
        l = p['results']['mean_latency_ms']
        t = p['results']['mean_tokens_per_second']
        print(f"  * {th} threads -> Latency: {l:.2f} ms | Throughput: {t:.2f} tok/s")
        
    print("\nInterpretation:")
    print(f"On this x86_64 development machine, the benchmark found {best_th} CPU threads to be the fastest tested configuration.")
    print("Note: x86_64 development measurements do NOT represent Arm64 cloud performance.")
    
if __name__ == "__main__":
    main()
