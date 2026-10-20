import json

with open("results/results_benchmark_10seeds.json", "r") as f:
    d_low = json.load(f)

with open("results/results_benchmark_full_10seeds.json", "r") as f:
    d_full = json.load(f)

print("=" * 105)
print("DEEP COMPARATIVE ANALYSIS: LOW-DATA REGIME (10%) vs. FULL-DATA REGIME (100%)")
print("=" * 105)
print(f"{'MODEL':<25} | {'LOW-DATA ACC':<18} | {'FULL-DATA ACC':<18} | {'DELTA ACC':<12} | {'LOW ROC':<10} | {'FULL ROC':<10}")
print("-" * 105)

for m in d_low["aggregated_metrics"].keys():
    acc_low = d_low["aggregated_metrics"][m]["accuracy"]["mean"]
    std_low = d_low["aggregated_metrics"][m]["accuracy"]["std"]
    acc_full = d_full["aggregated_metrics"][m]["accuracy"]["mean"]
    std_full = d_full["aggregated_metrics"][m]["accuracy"]["std"]
    delta_acc = acc_full - acc_low
    roc_low = d_low["aggregated_metrics"][m]["roc_auc"]["mean"]
    roc_full = d_full["aggregated_metrics"][m]["roc_auc"]["mean"]
    print(f"{m:<25} | {acc_low:.4f} ± {std_low:.4f}     | {acc_full:.4f} ± {std_full:.4f}      | {delta_acc:+7.2%}    | {roc_low:.4f}    | {roc_full:.4f}")

print("=" * 105)
