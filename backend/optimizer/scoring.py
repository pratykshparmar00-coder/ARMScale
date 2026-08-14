from typing import List, Dict, Any
from .models import Objective

class ScoringEngine:
    
    @staticmethod
    def _normalize(val: float, min_val: float, max_val: float, invert: bool = False) -> float:
        """
        Normalizes a value between 0 and 1.
        If invert is True, lower original values give higher scores (e.g. latency).
        """
        if max_val == min_val:
            return 1.0
        norm = (val - min_val) / (max_val - min_val)
        return 1.0 - norm if invert else norm

    @staticmethod
    def score_results(results: List[Dict[str, Any]], objective: Objective) -> Dict[str, Any]:
        """
        Assigns a score to each result based on the chosen objective.
        Sorts the list so the best configuration is first.
        """
        if not results:
            return None
            
        latencies = [r['results']['mean_latency_ms'] for r in results]
        tps = [r['results']['mean_tokens_per_second'] for r in results]
        # Since memory isn't fully profiled in Python without heavy tracing, we use placeholder or simple memory metric if provided
        # We will use 0 for memory if not captured, but we must implement the logic.
        memories = [r['results'].get('memory_mb', 0) for r in results]
        
        min_lat, max_lat = min(latencies), max(latencies)
        min_tps, max_tps = min(tps), max(tps)
        min_mem, max_mem = min(memories), max(memories)

        for r in results:
            res = r['results']
            l = res['mean_latency_ms']
            t = res['mean_tokens_per_second']
            m = res.get('memory_mb', 0)
            
            l_score = ScoringEngine._normalize(l, min_lat, max_lat, invert=True)
            t_score = ScoringEngine._normalize(t, min_tps, max_tps, invert=False)
            m_score = ScoringEngine._normalize(m, min_mem, max_mem, invert=True)
            
            if objective == Objective.SPEED:
                score = l_score * 0.9 + t_score * 0.1
            elif objective == Objective.THROUGHPUT:
                score = t_score * 0.9 + l_score * 0.1
            elif objective == Objective.MEMORY:
                score = m_score * 0.9 + l_score * 0.1
            else: # BALANCED
                score = (l_score + t_score + m_score) / 3.0
                
            r['score'] = score
            
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[0]

    @staticmethod
    def get_pareto_frontier(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        A configuration is Pareto-dominated if another configuration is:
        - no worse in all selected metrics
        - strictly better in at least one metric
        Metrics evaluated: mean_latency_ms (lower is better), mean_tokens_per_second (higher is better)
        """
        pareto = []
        for i, r1 in enumerate(results):
            is_dominated = False
            lat1 = r1['results']['mean_latency_ms']
            tps1 = r1['results']['mean_tokens_per_second']
            
            for j, r2 in enumerate(results):
                if i == j: continue
                lat2 = r2['results']['mean_latency_ms']
                tps2 = r2['results']['mean_tokens_per_second']
                
                # Check if r2 dominates r1
                better_or_equal = (lat2 <= lat1) and (tps2 >= tps1)
                strictly_better = (lat2 < lat1) or (tps2 > tps1)
                
                if better_or_equal and strictly_better:
                    is_dominated = True
                    break
                    
            if not is_dominated:
                pareto.append(r1)
                
        return pareto
