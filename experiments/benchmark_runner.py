import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import time
import numpy as np
import torch
from src.data import get_octmnist_loaders
from experiments.trainer import train_single_model
from src.metrics import compute_statistical_comparison, format_significance_stars
from src.gating_analysis import plot_class_routing_bar, plot_entropy_trajectory

DEFAULT_SEEDS = [0, 42, 100, 2023, 777, 999, 1234, 5678, 1111, 2222]
ALL_MODELS = [
    "proposed_dual_qnode",
    "symmetrical_classical",
    "dual_qnode_concat",
    "dual_qnode_fixed_sum",
    "single_qnode_z",
    "single_qnode_x",
    "pure_classical"
]

def run_benchmark(models=None, seeds=None, regime="low_data", epochs=20, batch_size=128, num_classes=4, output_json=None):
    if models is None:
        models = ALL_MODELS
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if output_json is None:
        output_json = f"results/results_benchmark_{regime}_10seeds.json"
        
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Fully unlocked hardware configuration: Max CPU threads + GPU CUDA
    # T2_THREADS env var caps threads when multiple workers share the machine.
    num_threads = int(os.environ.get('T2_THREADS', '0')) or (os.cpu_count() or 8)
    torch.set_num_threads(num_threads)
    dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print(f"ADAPTIVE QUANTUM GATING BENCHMARK ({regime.upper()} REGIME)")
    print("=" * 80)
    print(f"[*] Compute Device:      {dev_str.upper()} (GPU Acceleration: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"[*] GPU Name:            {torch.cuda.get_device_name(0)}")
    print(f"[*] CPU Threads:         {num_threads} (Fully Unlocked Maximum Parallelism)")
    print(f"[*] Models to evaluate:  {len(models)} -> {models}")
    print(f"[*] Seeds to evaluate:   {len(seeds)} -> {seeds}")
    print(f"[*] Regime:              {regime} (4 Classes: CNV, DME, DRUSEN, NORMAL)")
    print(f"[*] Epochs per run:      {epochs} | Batch size: {batch_size}")
    print("=" * 80)
    
    raw_results = {m: [] for m in models}
    total_runs = len(models) * len(seeds)
    run_idx = 0
    t_start = time.time()
    
    for seed in seeds:
        print(f"\n[>>>] Initializing DataLoaders for Seed {seed} (Regime: {regime})...")
        tr_loader, val_loader, te_loader, class_weights, _, _ = get_octmnist_loaders(
            batch_size=batch_size,
            regime=regime,
            seed=seed,
            num_classes=num_classes
        )
        for m_name in models:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] Running {m_name:<25} | Seed: {seed} | Epochs: {epochs} | Dev: {dev_str.upper()}")
            res = train_single_model(
                model_name=m_name,
                train_loader=tr_loader,
                val_loader=val_loader,
                test_loader=te_loader,
                class_weights=class_weights,
                seed=seed,
                epochs=epochs,
                num_classes=num_classes,
                device=dev_str
            )
            raw_results[m_name].append(res)
            test_m = res["test_metrics"]
            print(f"    -> Acc: {test_m['accuracy']:.4f} | B-Acc: {test_m['balanced_accuracy']:.4f} | F1: {test_m['f1_macro']:.4f} | MCC: {test_m['mcc']:.4f} | ROC-AUC: {test_m['roc_auc']:.4f} | PR-AUC: {test_m['pr_auc']:.4f} ({res['training_time_sec']:.1f}s)")
            
    metrics_list = ["accuracy", "balanced_accuracy", "f1_macro", "mcc", "roc_auc", "pr_auc"]
    aggregated = {}
    for m_name in models:
        aggregated[m_name] = {}
        for met in metrics_list:
            vals = [r["test_metrics"][met] for r in raw_results[m_name]]
            aggregated[m_name][met] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "values": [float(v) for v in vals]
            }
            
    stat_comparisons = {}
    if "proposed_dual_qnode" in raw_results:
        prop_runs = raw_results["proposed_dual_qnode"]
        for other_m in models:
            if other_m == "proposed_dual_qnode":
                continue
            other_runs = raw_results[other_m]
            stat_comparisons["proposed_vs_" + other_m] = {}
            for met in metrics_list:
                prop_vals = [r["test_metrics"][met] for r in prop_runs]
                other_vals = [r["test_metrics"][met] for r in other_runs]
                st = compute_statistical_comparison(prop_vals, other_vals)
                stat_comparisons["proposed_vs_" + other_m][met] = st
                
    gating_summary_all_seeds = {}
    if "proposed_dual_qnode" in raw_results:
        first_res = raw_results["proposed_dual_qnode"][0]
        if first_res.get("gating_analysis") is not None:
            gating_summary_all_seeds["seed_0_routing"] = first_res["gating_analysis"]
            plot_class_routing_bar(first_res["gating_analysis"], f"plots/fig_gating_class_routing_{regime}.png")
            plot_entropy_trajectory(first_res["history"]["epoch_entropy"], f"plots/fig_gating_entropy_trajectory_{regime}.png")
            
    full_output = {
        "metadata": {
            "benchmark_title": "Adaptive Gating of Complementary Variational Quantum Circuits",
            "dataset": f"OCTMNIST ({regime} regime)",
            "regime": regime,
            "num_seeds": len(seeds),
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": dev_str,
            "total_runtime_seconds": float(time.time() - t_start),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "aggregated_metrics": aggregated,
        "statistical_comparisons": stat_comparisons,
        "gating_dynamics": gating_summary_all_seeds,
        "raw_runs": raw_results
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n[OK] Benchmark completed! Results saved to: {output_json}")
    return full_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adaptive Quantum Gating Benchmark")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--regime", type=str, default="full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()
    
    model_list = args.models.split(",") if args.models else None
    seed_list = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else None
    
    run_benchmark(
        models=model_list,
        seeds=seed_list,
        regime=args.regime,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        output_json=args.output_json
    )
