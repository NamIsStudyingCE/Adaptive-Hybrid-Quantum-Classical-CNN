import json
import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("plots", exist_ok=True)
os.makedirs("GD_FULL_DATA", exist_ok=True)

with open("results/results_benchmark_10seeds.json", "r") as f:
    d_low = json.load(f)
with open("results/results_benchmark_full_10seeds.json", "r") as f:
    d_full = json.load(f)

models_order = [
    "proposed_dual_qnode",
    "symmetrical_classical",
    "dual_qnode_concat",
    "dual_qnode_fixed_sum",
    "single_qnode_z",
    "pure_classical"
]
labels = [
    "Proposed Dual-QNode",
    "Symmetrical Classical",
    "Dual Concat",
    "Dual Fixed Sum",
    "Single QNode-Z",
    "Pure Classical"
]

low_means = [d_low["aggregated_metrics"][m]["accuracy"]["mean"] * 100 for m in models_order]
low_stds = [d_low["aggregated_metrics"][m]["accuracy"]["std"] * 100 for m in models_order]

full_means = [d_full["aggregated_metrics"][m]["accuracy"]["mean"] * 100 for m in models_order]
full_stds = [d_full["aggregated_metrics"][m]["accuracy"]["std"] * 100 for m in models_order]

x = np.arange(len(models_order))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
rects1 = ax.bar(x - width/2, low_means, width, yerr=low_stds, label='Low-Data Regime (10% Data)', color='#2b5c8f', capsize=5, alpha=0.9)
rects2 = ax.bar(x + width/2, full_means, width, yerr=full_stds, label='Full-Data Regime (100% Data)', color='#d95f02', capsize=5, alpha=0.9)

ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Data Scaling Comparison: Low-Data (10%) vs. Full-Data (100%) on OCTMNIST', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=10, fontweight='medium')
ax.legend(loc='lower right', frameon=True, framealpha=0.9)
ax.set_ylim(60, 78)
ax.grid(axis='y', linestyle='--', alpha=0.5)

for i, rect in enumerate(rects1):
    height = rect.get_height()
    err = low_stds[i]
    ax.annotate(f'{height:.2f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height + err),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

for i, rect in enumerate(rects2):
    height = rect.get_height()
    err = full_stds[i]
    ax.annotate(f'{height:.2f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height + err),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("plots/fig_regime_comparison_bar.png", bbox_inches='tight')
plt.close()
print("[OK] Saved plots/fig_regime_comparison_bar.png")
