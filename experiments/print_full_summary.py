import json

with open("results/results_benchmark_full_10seeds.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("=" * 90)
print(f"{'MODEL':<26} | {'ACCURACY':<17} | {'BAL-ACC':<17} | {'F1-MACRO':<17} | {'ROC-AUC':<17}")
print("=" * 90)
for m, mets in d["aggregated_metrics"].items():
    acc_str = f"{mets['accuracy']['mean']:.4f} ± {mets['accuracy']['std']:.4f}"
    bacc_str = f"{mets['balanced_accuracy']['mean']:.4f} ± {mets['balanced_accuracy']['std']:.4f}"
    f1_str = f"{mets['f1_macro']['mean']:.4f} ± {mets['f1_macro']['std']:.4f}"
    roc_str = f"{mets['roc_auc']['mean']:.4f} ± {mets['roc_auc']['std']:.4f}"
    print(f"{m:<26} | {acc_str:<17} | {bacc_str:<17} | {f1_str:<17} | {roc_str:<17}")

print("\n" + "=" * 90)
print("STATISTICAL COMPARISONS (Proposed Dual-QNode Gating vs. Symmetrical Classical):")
print("=" * 90)
stat = d["statistical_comparisons"]["proposed_vs_symmetrical_classical"]
for k, v in stat.items():
    ci_str = f"[{v['ci_95'][0]:.4f}, {v['ci_95'][1]:.4f}]"
    print(f"{k:<18} | Diff: {v['mean_diff']:+.4f} | 95% CI: {ci_str:<20} | t-test p: {v['t_p_value']:.4f} | Wilcoxon p: {v['wilcoxon_p_value']:.4f} | Cohen's d: {v['cohens_d']:.4f}")
