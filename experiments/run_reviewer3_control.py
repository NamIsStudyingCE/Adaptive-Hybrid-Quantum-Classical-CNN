import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import torch
from scipy import stats
from src.data import get_octmnist_loaders
from experiments.trainer import train_single_model
from src.metrics import compute_statistical_comparison

SEEDS = [0, 42, 100, 2023, 777, 999, 1234, 5678, 1111, 2222]
MODEL_NAME = "cosine_bottleneck"
REGIME = "low_data"
EPOCHS = 20
BATCH_SIZE = 128
NUM_CLASSES = 4
OUTPUT_JSON = "results/results_cosine_bottleneck_low_data.json"

dev_str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Running Reviewer 3 Control on {dev_str.upper()}...")

raw_results = []
for seed in SEEDS:
    print(f"\n[>>>] Running Seed {seed}...")
    tr_loader, val_loader, te_loader, class_weights, _, _ = get_octmnist_loaders(
        batch_size=BATCH_SIZE, regime=REGIME, seed=seed, num_classes=NUM_CLASSES
    )
    res = train_single_model(
        model_name=MODEL_NAME,
        train_loader=tr_loader,
        val_loader=val_loader,
        test_loader=te_loader,
        class_weights=class_weights,
        seed=seed,
        epochs=EPOCHS,
        num_classes=NUM_CLASSES,
        device=dev_str
    )
    raw_results.append(res)
    tm = res["test_metrics"]
    print(f"  -> Acc: {tm['accuracy']:.4f} | B-Acc: {tm['balanced_accuracy']:.4f} | F1: {tm['f1_macro']:.4f} | ROC-AUC: {tm['roc_auc']:.4f}")

metrics_list = ["accuracy", "balanced_accuracy", "f1_macro", "mcc", "roc_auc", "pr_auc"]
aggregated = {}
for met in metrics_list:
    vals = [r["test_metrics"][met] for r in raw_results]
    aggregated[met] = {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "values": [float(v) for v in vals]
    }

# So sánh paired statistical tests với benchmark 10 seeds cũ
with open("results/results_benchmark_10seeds.json", "r", encoding="utf-8") as f:
    old_data = json.load(f)

old_z = old_data["raw_runs"]["single_qnode_z"]
old_prop = old_data["raw_runs"]["proposed_dual_qnode"]

cos_acc = aggregated["accuracy"]["values"]
z_acc = [r["test_metrics"]["accuracy"] for r in old_z]
prop_acc = [r["test_metrics"]["accuracy"] for r in old_prop]

t_z = stats.ttest_rel(cos_acc, z_acc)
w_z = stats.wilcoxon(np.array(cos_acc) - np.array(z_acc))
t_prop = stats.ttest_rel(cos_acc, prop_acc)
w_prop = stats.wilcoxon(np.array(cos_acc) - np.array(prop_acc))

# Detailed stats across all metrics
detailed_vs_z = {}
detailed_vs_prop = {}
for met in metrics_list:
    c_vals = np.array(aggregated[met]["values"])
    z_vals = np.array([r["test_metrics"][met] for r in old_z])
    p_vals = np.array([r["test_metrics"][met] for r in old_prop])
    detailed_vs_z[met] = compute_statistical_comparison(c_vals.tolist(), z_vals.tolist())
    detailed_vs_prop[met] = compute_statistical_comparison(c_vals.tolist(), p_vals.tolist())
    # TOST +-1%
    diff_z = c_vals - z_vals
    p_tost_z = float(max(
        stats.ttest_1samp(diff_z, -0.01, alternative="greater").pvalue,
        stats.ttest_1samp(diff_z, +0.01, alternative="less").pvalue
    ))
    diff_p = c_vals - p_vals
    p_tost_p = float(max(
        stats.ttest_1samp(diff_p, -0.01, alternative="greater").pvalue,
        stats.ttest_1samp(diff_p, +0.01, alternative="less").pvalue
    ))
    detailed_vs_z[met]["p_tost_1pct"] = p_tost_z
    detailed_vs_prop[met]["p_tost_1pct"] = p_tost_p

out_payload = {
    "metadata": {
        "model": MODEL_NAME,
        "n_params": raw_results[0]["num_params"],
        "regime": REGIME,
        "seeds": SEEDS,
        "epochs": EPOCHS
    },
    "aggregated_metrics": aggregated,
    "comparisons": {
        "vs_single_qnode_z": {
            "t_stat": float(t_z.statistic), "p_val": float(t_z.pvalue),
            "wilcoxon_stat": float(w_z.statistic), "wilcoxon_pval": float(w_z.pvalue),
            "all_metrics": detailed_vs_z
        },
        "vs_proposed_dual_qnode": {
            "t_stat": float(t_prop.statistic), "p_val": float(t_prop.pvalue),
            "wilcoxon_stat": float(w_prop.statistic), "wilcoxon_pval": float(w_prop.pvalue),
            "all_metrics": detailed_vs_prop
        }
    },
    "raw_runs": raw_results
}

os.makedirs("results", exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out_payload, f, indent=2)

print("\n" + "="*80)
print("EXPERIMENT COMPLETED SUCCESSFULLY.")
print(f"Cosine Control Accuracy: {aggregated['accuracy']['mean']:.4f} +- {aggregated['accuracy']['std']:.4f}")
print(f"F1 Macro:                {aggregated['f1_macro']['mean']:.4f} +- {aggregated['f1_macro']['std']:.4f}")
print(f"MCC:                     {aggregated['mcc']['mean']:.4f} +- {aggregated['mcc']['std']:.4f}")
print(f"ROC-AUC:                 {aggregated['roc_auc']['mean']:.4f} +- {aggregated['roc_auc']['std']:.4f}")
print(f"PR-AUC:                  {aggregated['pr_auc']['mean']:.4f} +- {aggregated['pr_auc']['std']:.4f}")
print(f"Comparison vs Single-Z:  t={t_z.statistic:.3f}, p={t_z.pvalue:.4f}, W={w_z.statistic:.1f}, p_w={w_z.pvalue:.4f}")
print(f"Comparison vs Proposed:  t={t_prop.statistic:.3f}, p={t_prop.pvalue:.4f}, W={w_prop.statistic:.1f}, p_w={w_prop.pvalue:.4f}")
print("="*80)

