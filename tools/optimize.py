import argparse
import time
from backend.optimizer.models import Objective, OptimizationDimension, OptimizationRequest
from backend.optimizer.engine import OptimizationEngine
from backend.api.main import engine, benchmark_engine
from backend.platform.detector import get_platform

def print_table(results):
    headers = [
        "CONFIG", "MEAN (ms)", "MEDIAN (ms)", "P95 (ms)", "MIN (ms)", "MAX (ms)", "STD (ms)", "TOK/S", "MEMORY", "SCORE"
    ]
    header_fmt = "{:<16} {:<12} {:<12} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<8}"
    print("\n" + header_fmt.format(*headers))
    print("-" * 114)
    
    for r in results:
        th = r['configuration']['threads']
        ctx = r['configuration']['context_size']
        cfg = f"T={th}, Ctx={ctx}"
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
    parser = argparse.ArgumentParser(description="ARMScale Optimization CLI (Platform-Agnostic Engine)")
    parser.add_argument("--objective", type=str, default="speed", choices=[e.value for e in Objective], help="Optimization objective (speed, throughput, balanced, memory)")
    parser.add_argument("--dimension", type=str, default="threads", choices=[e.value for e in OptimizationDimension], help="Target dimension (threads, context, combined)")
    parser.add_argument("--workload", type=str, default="short_generation", choices=["short_generation", "context_stress"], help="Workload suite (short_generation, context_stress)")
    parser.add_argument("--threads", type=str, help="Comma-separated list of thread counts to test (e.g., 1,2,4,6,8,12)")
    parser.add_argument("--contexts", type=str, help="Comma-separated list of context sizes to test (e.g., 1024,2048,4096)")
    
    args = parser.parse_args()
    
    platform = get_platform()
    platform_info = platform.to_dict()
    
    req = OptimizationRequest(
        objective=Objective(args.objective),
        dimension=OptimizationDimension(args.dimension),
        workload_type=args.workload
    )
    if args.threads:
        req.threads_to_test = [int(x.strip()) for x in args.threads.split(",")]
    if args.contexts:
        req.context_sizes_to_test = [int(x.strip()) for x in args.contexts.split(",")]
        
    print(f"=== ARMScale Optimization Engine ===")
    print(f"Target Objective: {req.objective.value.upper()}")
    print(f"Dimension:        {req.dimension.value.upper()}")
    print(f"Workload:         {req.workload_type.upper()}")
    print(f"Platform:         {platform_info['provider'].upper()} ({platform_info['architecture']})")
    print(f"CPU:              {platform_info['cpu']} ({platform_info['physical_cores']} physical / {platform_info['logical_cores']} logical cores)\n")
    
    engine.load_model()
    
    optimizer = OptimizationEngine(engine, benchmark_engine)
    exp_id = optimizer.start_optimization(req)
    
    print(f"Started experiment {exp_id}. Running benchmark sweep...")
    result = optimizer.run_optimization_sync(req, exp_id)
    
    print(f"\nOptimization Sweep Complete for Workload [{req.workload_type.upper()}]!\n")
    print_table(result.results)
    
    best = result.best_configuration
    best_cfg_str = f"Threads={best['configuration']['threads']}, Context={best['configuration']['context_size']}"
    print(f"WINNER ({req.objective.value.upper()}): {best_cfg_str}")
    
    base_res = result.baseline['results']
    base_cfg = result.baseline['configuration']
    print(f"\n--- Baseline Reference (Threads: {base_cfg['threads']}, Context: {base_cfg['context_size']}) ---")
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
        ctx = p['configuration']['context_size']
        l = p['results']['mean_latency_ms']
        t = p['results']['mean_tokens_per_second']
        print(f"  * Threads={th}, Context={ctx} -> Latency: {l:.2f} ms | Throughput: {t:.2f} tok/s")
        
    print("\nInterpretation:")
    print(f"On this {platform_info['architecture']} machine ({platform_info['provider']}), the benchmark found {best_cfg_str} to be the optimal tested configuration for workload '{req.workload_type}'.")
    if platform_info['architecture'] in ['amd64', 'x86_64']:
        print("Note: x86_64 development measurements do NOT represent Arm64 cloud performance.")
    
if __name__ == "__main__":
    main()
