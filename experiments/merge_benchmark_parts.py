import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
import numpy as np
from src.metrics import compute_statistical_comparison
from src.gating_analysis import plot_class_routing_bar, plot_entropy_trajectory

ALL_MODELS = [
    "proposed_dual_qnode",
    "symmetrical_classical",
    "dual_qnode_concat",
    "dual_qnode_fixed_sum",
    "single_qnode_z",
    "single_qnode_x",
    "pure_classical"
]
EXPECTED_SEEDS = [0, 42, 100, 2023, 777, 999, 1234, 5678, 1111, 2222]
METRICS_LIST = ["accuracy", "balanced_accuracy", "f1_macro", "mcc", "roc_auc", "pr_auc"]

def merge_partitions(parts_dir="results/parts", output_json="results/results_benchmark_full_10seeds.json", regime="full"):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    part_files = sorted([os.path.join(parts_dir, f) for f in os.listdir(parts_dir) if f.endswith(".json")])
    if not part_files:
        raise FileNotFoundError(f"No partition JSON files found in {parts_dir}")
        
    print(f"[*] Found {len(part_files)} partition files: {part_files}")
    
    combined_raw_runs = {m: [] for m in ALL_MODELS}
    collected_seeds = set()
    total_runtime = 0.0
    
    for pf in part_files:
        with open(pf, "r", encoding="utf-8") as f:
            data = json.load(f)
        seeds_in_part = data["metadata"]["seeds"]
        collected_seeds.update(seeds_in_part)
        total_runtime += data["metadata"].get("total_runtime_seconds", 0.0)
        
        for m_name in ALL_MODELS:
            if m_name in data["raw_runs"]:
                for run_dict in data["raw_runs"][m_name]:
                    # check if seed already present
                    if not any(r["seed"] == run_dict["seed"] for r in combined_raw_runs[m_name]):
                        combined_raw_runs[m_name].append(run_dict)
                        
    # Sort runs by seed index according to EXPECTED_SEEDS
    for m_name in ALL_MODELS:
        combined_raw_runs[m_name].sort(key=lambda r: EXPECTED_SEEDS.index(r["seed"]) if r["seed"] in EXPECTED_SEEDS else r["seed"])
        
    print(f"[*] Collected seeds ({len(collected_seeds)}/10): {sorted(list(collected_seeds))}")
    for m in ALL_MODELS:
        print(f"    - {m:<25}: {len(combined_raw_runs[m])} runs")
        
    # Aggregate Metrics
    aggregated = {}
    for m_name in ALL_MODELS:
        aggregated[m_name] = {}
        for met in METRICS_LIST:
            vals = [r["test_metrics"][met] for r in combined_raw_runs[m_name]]
            aggregated[m_name][met] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "values": [float(v) for v in vals]
            }
            
    # Statistical Comparisons
    stat_comparisons = {}
    if "proposed_dual_qnode" in combined_raw_runs:
        prop_runs = combined_raw_runs["proposed_dual_qnode"]
        for other_m in ALL_MODELS:
            if other_m == "proposed_dual_qnode":
                continue
            other_runs = combined_raw_runs[other_m]
            stat_comparisons["proposed_vs_" + other_m] = {}
            for met in METRICS_LIST:
                prop_vals = [r["test_metrics"][met] for r in prop_runs]
                other_vals = [r["test_metrics"][met] for r in other_runs]
                st = compute_statistical_comparison(prop_vals, other_vals)
                stat_comparisons["proposed_vs_" + other_m][met] = st
                
    gating_summary = {}
    if "proposed_dual_qnode" in combined_raw_runs and len(combined_raw_runs["proposed_dual_qnode"]) > 0:
        first_res = combined_raw_runs["proposed_dual_qnode"][0]
        if first_res.get("gating_analysis") is not None:
            gating_summary["seed_0_routing"] = first_res["gating_analysis"]
            plot_class_routing_bar(first_res["gating_analysis"], f"plots/fig_gating_class_routing_{regime}.png")
            plot_entropy_trajectory(first_res["history"]["epoch_entropy"], f"plots/fig_gating_entropy_trajectory_{regime}.png")
            
    full_output = {
        "metadata": {
            "benchmark_title": "Adaptive Gating of Complementary Variational Quantum Circuits",
            "dataset": f"OCTMNIST ({regime} regime)",
            "regime": regime,
            "num_seeds": len(collected_seeds),
            "seeds": [s for s in EXPECTED_SEEDS if s in collected_seeds],
            "epochs": 20,
            "batch_size": 128,
            "device": "CUDA (Multi-Worker Parallel)",
            "total_runtime_seconds": float(total_runtime),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "aggregated_metrics": aggregated,
        "statistical_comparisons": stat_comparisons,
        "gating_dynamics": gating_summary,
        "raw_runs": combined_raw_runs
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n[OK] Successfully merged all partitions! Saved to: {output_json}")
    return full_output

if __name__ == "__main__":
    merge_partitions()
