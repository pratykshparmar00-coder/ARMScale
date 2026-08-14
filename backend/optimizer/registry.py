import os
import glob
import json
from typing import List, Dict, Any, Optional

class ExperimentRegistry:
    """Discovers and catalogs completed optimization experiments across standard and global runs."""

    def __init__(self, results_dir: Optional[str] = None):
        if results_dir:
            self.results_dir = results_dir
        else:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.results_dir = os.path.join(root_dir, "benchmarks", "results", "optimization")

    def _get_all_json_files(self) -> List[str]:
        if not os.path.exists(self.results_dir):
            return []
        pattern = os.path.join(self.results_dir, "**", "optimization_*.json")
        files = glob.glob(pattern, recursive=True)
        # Sort by modification time descending (newest first)
        return sorted(files, key=os.path.getmtime, reverse=True)

    def list_experiments(self, workload_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns metadata summaries for all recorded experiments without duplication."""
        json_files = self._get_all_json_files()
        experiments = []
        seen_ids = set()
        
        for fpath in json_files:
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    exp_id = data.get("experiment_id")
                    if exp_id and exp_id in seen_ids:
                        continue
                    if exp_id:
                        seen_ids.add(exp_id)
                        
                    w_type = data.get("workload_type", data.get("workload", {}).get("type", "unknown"))
                    if workload_type and w_type != workload_type:
                        continue
                        
                    best_cfg = data.get("best_configuration", {})
                    cfg = best_cfg.get("configuration", {}) if isinstance(best_cfg, dict) else {}
                    
                    experiments.append({
                        "experiment_id": exp_id,
                        "timestamp": data.get("timestamp", os.path.basename(fpath)),
                        "dimension": data.get("dimension", "unknown"),
                        "workload_type": w_type,
                        "platform": data.get("platform", {}).get("provider", "local"),
                        "architecture": data.get("platform", {}).get("architecture", "unknown"),
                        "configurations_tested": data.get("configurations_tested", len(data.get("results", []))),
                        "best_configuration": cfg,
                        "pareto_count": len(data.get("pareto_configurations", [])),
                        "artifact_path": fpath
                    })
            except Exception as e:
                print(f"Warning: Failed to parse experiment {fpath}: {e}")
                
        return experiments

    def get_latest_experiment(self, workload_type: Optional[str] = None, dimension: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves the full JSON payload of the most recent experiment matching filters."""
        json_files = self._get_all_json_files()
        for fpath in json_files:
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    w_type = data.get("workload_type", data.get("workload", {}).get("type"))
                    d_type = data.get("dimension")
                    if workload_type and w_type != workload_type:
                        continue
                    if dimension and d_type != dimension:
                        continue
                    return data
            except Exception:
                continue
        return None
