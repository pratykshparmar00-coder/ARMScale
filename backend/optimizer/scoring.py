from typing import List, Dict, Any, Optional
from .models import Objective

class ScoringEngine:
    
    @staticmethod
    def _normalize(val: float, min_val: float, max_val: float, invert: bool = False) -> float:
        """
        Normalizes a value between 0 and 1.
        If invert is True, lower original values yield higher normalized scores (desirable for latency/memory).
        If min_val == max_val, returns 1.0 (all configurations tied).
        """
        if max_val == min_val:
            return 1.0
        norm = (val - min_val) / (max_val - min_val)
        return 1.0 - norm if invert else norm

    @staticmethod
    def score_results(results: List[Dict[str, Any]], objective: Objective) -> Optional[Dict[str, Any]]:
        """
        Assigns a normalized score [0.0, 1.0] to each result based on the chosen objective.
        Sorts the list in-place in descending order of score, returning the top configuration.

        Scoring Formulas:
        - SPEED: 90% normalized inverse latency + 10% normalized throughput
          score = (latency_score * 0.9) + (throughput_score * 0.1)
        - THROUGHPUT: 90% normalized throughput + 10% normalized inverse latency
          score = (throughput_score * 0.9) + (latency_score * 0.1)
        - BALANCED: 50% normalized inverse latency + 50% normalized throughput (since memory is unavailable)
          score = (latency_score + throughput_score) / 2.0
        - MEMORY: If memory measurement is unavailable, falls back to BALANCED with a documented note.
        """
        if not results:
            return None
            
        latencies = [r['results']['mean_latency_ms'] for r in results]
        tps = [r['results']['mean_tokens_per_second'] for r in results]
        
        min_lat, max_lat = min(latencies), max(latencies)
        min_tps, max_tps = min(tps), max(tps)

        for r in results:
            res = r['results']
            l = res['mean_latency_ms']
            t = res['mean_tokens_per_second']
            
            l_score = ScoringEngine._normalize(l, min_lat, max_lat, invert=True)
            t_score = ScoringEngine._normalize(t, min_tps, max_tps, invert=False)
            
            if objective == Objective.SPEED:
                score = (l_score * 0.9) + (t_score * 0.1)
            elif objective == Objective.THROUGHPUT:
                score = (t_score * 0.9) + (l_score * 0.1)
            elif objective == Objective.BALANCED:
                score = (l_score + t_score) / 2.0
            elif objective == Objective.MEMORY:
                score = (l_score + t_score) / 2.0
                r['scoring_note'] = "Memory optimization is deferred until native/process-level measurement is implemented; balanced scoring applied."
            else:
                score = (l_score + t_score) / 2.0
                
            r['score'] = score
            
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[0]

    @staticmethod
    def get_pareto_frontier(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculates the Pareto frontier from tested configurations.
        A configuration A dominates B if:
          (A.latency <= B.latency AND A.throughput >= B.throughput)
          AND (A.latency < B.latency OR A.throughput > B.throughput)
          
        Also annotates each item in `results` with `pareto_optimal: True/False`.
        Returns all non-dominated configurations.
        """
        pareto = []
        for i, r1 in enumerate(results):
            is_dominated = False
            lat1 = r1['results']['mean_latency_ms']
            tps1 = r1['results']['mean_tokens_per_second']
            
            for j, r2 in enumerate(results):
                if i == j:
                    continue
                lat2 = r2['results']['mean_latency_ms']
                tps2 = r2['results']['mean_tokens_per_second']
                
                # Check if r2 strictly dominates r1
                better_or_equal = (lat2 <= lat1) and (tps2 >= tps1)
                strictly_better = (lat2 < lat1) or (tps2 > tps1)
                
                if better_or_equal and strictly_better:
                    is_dominated = True
                    break
                    
            r1["pareto_optimal"] = not is_dominated
            if not is_dominated:
                pareto.append(r1)
                
        return pareto
