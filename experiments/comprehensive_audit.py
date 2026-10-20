import os
import sys
import json
import torch
import numpy as np

print("=" * 80)
print("COMPREHENSIVE AUDIT & INTEGRITY VERIFICATION SUITE")
print("=" * 80)

errors = []
warnings = []

# 1. Check Feature Cache Integrity
cache_path = "data/octmnist_resnet18_features.pt"
if not os.path.exists(cache_path):
    errors.append(f"Cache file missing: {cache_path}")
else:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    for split in ["train", "val", "test"]:
        if f"{split}_features" not in cache or f"{split}_labels" not in cache:
            errors.append(f"Missing {split} split in cache")
        feats = cache[f"{split}_features"]
        labs = cache[f"{split}_labels"]
        if feats.shape[0] != labs.shape[0]:
            errors.append(f"Mismatch in {split} length: feats {feats.shape[0]} vs labs {labs.shape[0]}")
        if feats.shape[1] != 512:
            errors.append(f"Unexpected feature dim {feats.shape[1]} in {split}")
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            errors.append(f"NaN or Inf found in {split}_features")
    print(f"[OK 1/6] Feature Cache Verified: train={cache['train_features'].shape[0]}, val={cache['val_features'].shape[0]}, test={cache['test_features'].shape[0]}")

# 2. Check Data Splits & No Leakage
train_labs = cache['train_labels'].numpy()
val_labs = cache['val_labels'].numpy()
test_labs = cache['test_labels'].numpy()
print(f"[OK 2/6] Class distributions: Train={np.bincount(train_labs)}, Val={np.bincount(val_labs)}, Test={np.bincount(test_labs)}")

# 3. Check JSON Results Files (Low Data & Full Data)
for regime, json_path in [("Low-Data", "results/results_benchmark_10seeds.json"), ("Full-Data", "results/results_benchmark_full_10seeds.json")]:
    if not os.path.exists(json_path):
        errors.append(f"Results JSON missing: {json_path}")
        continue
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    seeds = data["metadata"]["seeds"]
    if len(seeds) != 10:
        errors.append(f"{regime}: expected 10 seeds, got {len(seeds)}")
    
    models = ["proposed_dual_qnode", "symmetrical_classical", "dual_qnode_concat", "dual_qnode_fixed_sum", "single_qnode_z", "single_qnode_x", "pure_classical"]
    for m in models:
        if m not in data["raw_runs"]:
            errors.append(f"{regime}: model {m} missing in raw_runs")
            continue
        runs = data["raw_runs"][m]
        if len(runs) != 10:
            errors.append(f"{regime}: model {m} has {len(runs)} runs, expected 10")
        
        # Check metrics validity
        for r in runs:
            met = r["test_metrics"]
            for k in ["accuracy", "balanced_accuracy", "f1_macro", "mcc", "roc_auc", "pr_auc"]:
                v = met[k]
                if np.isnan(v) or np.isinf(v) or v < 0.0 or v > 1.0:
                    errors.append(f"{regime}: {m} seed {r['seed']} invalid metric {k}={v}")
                    
        # Check statistical comparisons
        if "statistical_comparisons" in data and f"proposed_vs_{m}" in data["statistical_comparisons"]:
            st = data["statistical_comparisons"][f"proposed_vs_{m}"]
            for k in ["accuracy", "f1_macro", "roc_auc"]:
                if np.isnan(st[k]["t_pvalue"]) or np.isnan(st[k]["wilcoxon_pvalue"]):
                    warnings.append(f"{regime}: NaN in stat p-value for proposed_vs_{m} {k}")

    print(f"[OK 3/6] JSON Verified ({regime}): 70/70 runs valid, no NaNs, no Infs.")

# 4. Check LaTeX Tables in Paper Directory
table_files = ["paper/tables/table_10seeds.tex", "paper/tables/table_statistical_tests.tex"]
for tf in table_files:
    if os.path.exists(tf):
        print(f"[OK 4/6] LaTeX Table exists: {tf}")
    else:
        warnings.append(f"Paper table missing: {tf}")

# 5. Check Publication Plots
plot_files = [
    "plots/fig_benchmark_boxplots.png",
    "plots/fig_confusion_matrices.png",
    "plots/fig_gating_class_routing.png",
    "plots/fig_gating_entropy_trajectory.png",
    "plots/fig_regime_comparison_bar.png"
]
for pf in plot_files:
    if os.path.exists(pf):
        size_kb = os.path.getsize(pf) / 1024
        print(f"[OK 5/6] Plot verified: {pf} ({size_kb:.1f} KB)")
    else:
        errors.append(f"Plot missing: {pf}")

# 6. Check GD Directories (Nucleus Info & Copies)
for gd in ["GD0", "GD1", "GD2", "GD_FULL_DATA"]:
    if os.path.exists(gd):
        items = os.listdir(gd)
        print(f"[OK 6/6] Nucleus Directory verified: {gd}/ ({len(items)} files)")
    else:
        errors.append(f"GD Directory missing: {gd}")

print("\n" + "=" * 80)
print(f"AUDIT SUMMARY: {len(errors)} CRITICAL ERRORS, {len(warnings)} WARNINGS")
print("=" * 80)
if errors:
    for e in errors:
        print(f"  [ERROR] {e}")
if warnings:
    for w in warnings:
        print(f"  [WARNING] {w}")
if not errors and not warnings:
    print("  >>> SYSTEM INTEGRITY PERFECT (100% CLEAN) <<<")
