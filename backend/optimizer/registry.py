import os
import glob
import json
from typing import List, Dict, Any, Optional

class ExperimentRegistry:
    """Discovers and catalogs completed optimization experiments."""

    def __init__(self, results_dir: Optional[str] = None):
        if results_dir:
            self.results_dir = results_dir
        else:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.results_dir = os.path.join(root_dir, "benchmarks", "results", "optimization")

    def list_experiments(self, workload_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns metadata summaries for all recorded experiments."""
        if not os.path.exists(self.results_dir):
            return []
            
        json_files = sorted(glob.glob(os.path.join(self.results_dir, "optimization_*.json")), reverse=True)
        experiments = []
        
        for fpath in json_files:
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    w_type = data.get("workload_type", data.get("workload", {}).get("type", "unknown"))
                    if workload_type and w_type != workload_type:
                        continue
                        
                    experiments.append({
                        "experiment_id": data.get("experiment_id"),
                        "timestamp": data.get("timestamp", os.path.basename(fpath)),
                        "dimension": data.get("dimension", "unknown"),
                        "workload_type": w_type,
                        "platform": data.get("platform", {}).get("provider", "local"),
                        "architecture": data.get("platform", {}).get("architecture", "unknown"),
                        "configurations_tested": data.get("configurations_tested", len(data.get("results", []))),
                        "best_configuration": data.get("best_configuration", {}).get("configuration"),
                        "pareto_count": len(data.get("pareto_configurations", [])),
                        "artifact_path": fpath
                    })
            except Exception as e:
                print(f"Warning: Failed to parse experiment {fpath}: {e}")
                
        return experiments

    def get_latest_experiment(self, workload_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves the full JSON payload of the most recent experiment matching workload_type."""
        if not os.path.exists(self.results_dir):
            return None
            
        json_files = sorted(glob.glob(os.path.join(self.results_dir, "optimization_*.json")), reverse=True)
        for fpath in json_files:
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    w_type = data.get("workload_type", data.get("workload", {}).get("type"))
                    if workload_type and w_type != workload_type:
                        continue
                    return data
            except Exception:
                continue
        return None
