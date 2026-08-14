import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from backend.optimizer.models import Objective, OptimizationDimension, OptimizationRequest
from backend.optimizer.engine import OptimizationEngine
from backend.api.main import engine, benchmark_engine
from backend.platform.detector import get_platform

def print_table(results):
    headers = [
        "QUANT", "CONFIG", "SIZE(MB)", "LOAD(ms)", "MEAN(ms)", "MEDIAN(ms)", "P95(ms)", "TOK/S", "PARETO", "SCORE"
    ]
    header_fmt = "{:<8} {:<14} {:<10} {:<10} {:<11} {:<11} {:<10} {:<9} {:<10} {:<8}"
    print("\n" + header_fmt.format(*headers))
    print("-" * 105)
    
    for r in results:
        quant = r['configuration'].get('quantization', 'Q4_K_M')
        th = r['configuration']['threads']
        ctx = r['configuration']['context_size']
        cfg = f"T={th}, C={ctx}"
        res = r['results']
        size_mb = f"{res.get('model_size_mb', 0.0):.1f}"
        load_ms = f"{res.get('load_time_ms', 0.0):.0f}"
        mean_l = f"{res['mean_latency_ms']:.2f}"
        med_l = f"{res['median_latency_ms']:.2f}"
        p95_l = f"{res['p95_latency_ms']:.2f}"
        tps = f"{res['mean_tokens_per_second']:.2f}"
        pareto = "YES" if r.get('pareto_optimal', False) else "NO"
        score = f"{r.get('score', 0):.4f}"
        print(header_fmt.format(quant, cfg, size_mb, load_ms, mean_l, med_l, p95_l, tps, pareto, score))
    print()

def main():
    parser = argparse.ArgumentParser(description="ARMScale Optimization CLI (Platform-Agnostic Engine)")
    parser.add_argument("--objective", type=str, default="speed", choices=[e.value for e in Objective], help="Optimization objective (speed, throughput, balanced, memory)")
    parser.add_argument("--dimension", type=str, default="threads", choices=[e.value for e in OptimizationDimension], help="Target dimension (threads, context, quantization, combined)")
    parser.add_argument("--workload", type=str, default="short_generation", choices=["short_generation", "context_stress"], help="Workload suite (short_generation, context_stress)")
    parser.add_argument("--threads", type=str, help="Comma-separated list of thread counts to test (e.g., 2,4,6,8)")
    parser.add_argument("--contexts", type=str, help="Comma-separated list of context sizes to test (e.g., 1024,2048,4096)")
    parser.add_argument("--quantizations", type=str, help="Comma-separated list of quantization variants (e.g., Q4_K_M,Q5_K_M,Q8_0)")
    
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
    if args.quantizations:
        req.quantizations_to_test = [x.strip().upper() for x in args.quantizations.split(",")]
        
    print(f"=== ARMScale Optimization Engine ===")
    print(f"Target Objective: {req.objective.value.upper()}")
    print(f"Dimension:        {req.dimension.value.upper()}")
    print(f"Workload:         {req.workload_type.upper()}")
    print(f"Platform:         {platform_info['provider'].upper()} ({platform_info['architecture']})")
    print(f"Host CPU:         {platform_info['cpu']} ({platform_info['physical_cores']} physical / {platform_info['logical_cores']} logical cores)\n")
    
    engine.load_model()
    
    optimizer = OptimizationEngine(engine, benchmark_engine)
    exp_id = optimizer.start_optimization(req)
    
    print(f"Started experiment {exp_id}. Running benchmark sweep...")
    result = optimizer.run_optimization_sync(req, exp_id)
    
    print(f"\nOptimization Sweep Complete for Workload [{req.workload_type.upper()}]!\n")
    print_table(result.results)
    
    best = result.best_configuration
    best_q = best['configuration'].get('quantization', 'Q4_K_M')
    best_th = best['configuration']['threads']
    best_ctx = best['configuration']['context_size']
    best_cfg_str = f"Quant={best_q}, Threads={best_th}, Context={best_ctx}"
    print(f"WINNER ({req.objective.value.upper()}): {best_cfg_str}")
    
    base_res = result.baseline['results']
    base_cfg = result.baseline['configuration']
    base_q = base_cfg.get('quantization', 'Q4_K_M')
    print(f"\n--- Baseline Reference (Quant: {base_q}, Threads: {base_cfg['threads']}, Context: {base_cfg['context_size']}) ---")
    print(f"Baseline Mean Latency:    {base_res['mean_latency_ms']:.2f} ms")
    print(f"Baseline Mean Throughput: {base_res['mean_tokens_per_second']:.2f} tok/s")
    
    print(f"\n--- Improvement vs Baseline ---")
    print(f"Latency Improvement:    {result.improvement_vs_baseline['latency_pct']:.2f}%")
    print(f"Throughput Improvement: {result.improvement_vs_baseline['throughput_pct']:.2f}%")
    mem_imp = result.improvement_vs_baseline.get('memory_pct')
    print(f"Memory Improvement:     {f'{mem_imp:.2f}%' if mem_imp is not None else 'N/A (unavailable)'}\n")
    
    print("PARETO CONFIGURATIONS:")
    for p in result.pareto_configurations:
        q = p['configuration'].get('quantization', 'Q4_K_M')
        th = p['configuration']['threads']
        ctx = p['configuration']['context_size']
        size = p['results'].get('model_size_mb', 0.0)
        l = p['results']['mean_latency_ms']
        t = p['results']['mean_tokens_per_second']
        print(f"  * {q} (T={th}, C={ctx}, {size:.1f}MB) -> Latency: {l:.2f} ms | Throughput: {t:.2f} tok/s")
        
    print("\nInterpretation:")
    print(f"On this {platform_info['architecture']} machine ({platform_info['provider']}), the benchmark found {best_cfg_str} to be the optimal tested configuration for workload '{req.workload_type}'.")
    if platform_info['architecture'] in ['amd64', 'x86_64']:
        print("Note: x86_64 development measurements do NOT represent Arm64 cloud performance.")
    
if __name__ == "__main__":
    main()
