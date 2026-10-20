import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OCT_CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]

def plot_all_figures(json_path="results/results_benchmark_10seeds.json", plots_dir="plots"):
    if not os.path.exists(json_path):
        print(f"[!] JSON results not found at: {json_path}")
        return
    os.makedirs(plots_dir, exist_ok=True)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    raw_runs = data["raw_runs"]
    agg = data["aggregated_metrics"]
    
    # 1. Boxplots of 6 Metrics across Models
    metrics_info = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("f1_macro", "Macro F1-Score"),
        ("mcc", "Matthews Corr. Coef. (MCC)"),
        ("roc_auc", "ROC-AUC (OvR)"),
        ("pr_auc", "PR-AUC (OvR)")
    ]
    
    models = list(agg.keys())
    model_labels = [
        "Proposed Dual-QNode",
        "Symmetrical Classical",
        "Dual-QNode Concat",
        "Dual-QNode Fixed Sum",
        "Single QNode (Z)",
        "Single QNode (X)",
        "Pure Classical"
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
    axes = axes.flatten()
    
    for idx, (met_key, met_title) in enumerate(metrics_info):
        ax = axes[idx]
        data_to_plot = []
        for m in models:
            vals = [r["test_metrics"][met_key] for r in raw_runs[m]]
            data_to_plot.append(vals)
            
        bplot = ax.boxplot(
            data_to_plot,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            widths=0.6
        )
        for patch, color in zip(bplot["boxes"], colors[:len(models)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            
        ax.set_title(met_title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels([f"M{i+1}" for i in range(len(models))], fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.8, label=f"M{i+1}: {model_labels[i]}")
        for i in range(len(models))
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), fontsize=10, framealpha=0.9)
    fig.suptitle("Multi-Seed Metric Distributions Across Comparative Architectures (10 Independent Seeds)", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    
    box_path = os.path.join(plots_dir, "fig_benchmark_boxplots.png")
    fig.savefig(box_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved boxplot figure to: {box_path}")
    
    # 2. Confusion Matrices Comparison with pure Matplotlib
    if "proposed_dual_qnode" in raw_runs and "symmetrical_classical" in raw_runs:
        cm_prop = np.array(raw_runs["proposed_dual_qnode"][0]["confusion_matrix"])
        cm_base = np.array(raw_runs["symmetrical_classical"][0]["confusion_matrix"])
        
        cm_prop_norm = cm_prop.astype(float) / cm_prop.sum(axis=1)[:, np.newaxis]
        cm_base_norm = cm_base.astype(float) / cm_base.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
        
        # Plot 1: Proposed
        cax1 = ax1.matshow(cm_prop_norm, cmap=plt.cm.Blues, alpha=0.85)
        for i in range(cm_prop_norm.shape[0]):
            for j in range(cm_prop_norm.shape[1]):
                ax1.text(x=j, y=i, s=f"{cm_prop_norm[i, j]:.2f}\n({cm_prop[i, j]})", va='center', ha='center', fontsize=9, fontweight='bold', color='black' if cm_prop_norm[i, j] < 0.6 else 'white')
        ax1.set_title("Proposed Dual-QNode Gating", fontsize=11, fontweight="bold", pad=15)
        ax1.set_xlabel("Predicted Label", fontweight="bold")
        ax1.set_ylabel("True Label", fontweight="bold")
        ax1.set_xticks(range(4))
        ax1.set_yticks(range(4))
        ax1.set_xticklabels(OCT_CLASSES)
        ax1.set_yticklabels(OCT_CLASSES)
        
        # Plot 2: Classical Baseline
        cax2 = ax2.matshow(cm_base_norm, cmap=plt.cm.Oranges, alpha=0.85)
        for i in range(cm_base_norm.shape[0]):
            for j in range(cm_base_norm.shape[1]):
                ax2.text(x=j, y=i, s=f"{cm_base_norm[i, j]:.2f}\n({cm_base[i, j]})", va='center', ha='center', fontsize=9, fontweight='bold', color='black' if cm_base_norm[i, j] < 0.6 else 'white')
        ax2.set_title("Symmetrical Classical Baseline", fontsize=11, fontweight="bold", pad=15)
        ax2.set_xlabel("Predicted Label", fontweight="bold")
        ax2.set_ylabel("True Label", fontweight="bold")
        ax2.set_xticks(range(4))
        ax2.set_yticks(range(4))
        ax2.set_xticklabels(OCT_CLASSES)
        ax2.set_yticklabels(OCT_CLASSES)
        
        fig.tight_layout()
        cm_path = os.path.join(plots_dir, "fig_confusion_matrices.png")
        fig.savefig(cm_path, dpi=300)
        plt.close(fig)
        print(f"[OK] Saved confusion matrix figure to: {cm_path}")

if __name__ == "__main__":
    plot_all_figures()
