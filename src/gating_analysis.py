import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OCT_CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

def compute_gating_entropy(weights, eps=1e-12):
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    weights = np.clip(weights, eps, 1.0 - eps)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    entropy = -np.sum(weights * np.log2(weights), axis=-1)
    return entropy

def analyze_class_gating(weights, labels, num_classes=4, class_names=OCT_CLASSES):
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy().astype(int)
    entropy = compute_gating_entropy(weights)
    class_stats = {}
    for c in range(num_classes):
        mask = (labels == c)
        if np.sum(mask) > 0:
            w_c = weights[mask]
            h_c = entropy[mask]
            c_name = class_names[c] if c < len(class_names) else 'Class_' + str(c)
            class_stats[c_name] = {
                'mean_w1_z': float(np.mean(w_c[:, 0])),
                'mean_w2_x': float(np.mean(w_c[:, 1])),
                'std_w1': float(np.std(w_c[:, 0])),
                'std_w2': float(np.std(w_c[:, 1])),
                'mean_entropy': float(np.mean(h_c)),
                'count': int(np.sum(mask))
            }
    return class_stats

def plot_class_routing_bar(class_stats, output_path='plots/fig_gating_class_routing.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    c_names = list(class_stats.keys())
    w1_vals = [class_stats[c]['mean_w1_z'] for c in c_names]
    w2_vals = [class_stats[c]['mean_w2_x'] for c in c_names]
    x = np.arange(len(c_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.bar(x - width/2, w1_vals, width, label='Ansatz 1 (Pauli-Z: RY-CZ)', color='#1f77b4', alpha=0.9)
    ax.bar(x + width/2, w2_vals, width, label='Ansatz 2 (Pauli-X: RZ-CNOT)', color='#ff7f0e', alpha=0.9)
    ax.set_ylabel('Mean Gate Routing Weight')
    ax.set_title('Adaptive Gating Routing Preference across OCT Pathologies', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(c_names, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Uniform Baseline (50/50)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print('Saved class routing plot to: ' + output_path)

def plot_entropy_trajectory(epoch_entropies, output_path='plots/fig_gating_entropy_trajectory.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = np.arange(1, len(epoch_entropies) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)
    ax.plot(epochs, epoch_entropies, marker='o', color='#2ca02c', linewidth=2, label='Mean Gating Shannon Entropy H(w)')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label='Uniform Mixture (1.0 bit)')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Entropy (bits)', fontweight='bold')
    ax.set_title('Evolution of Gating Specialization across Training Epochs', fontsize=11, fontweight='bold')
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print('Saved entropy trajectory plot to: ' + output_path)
