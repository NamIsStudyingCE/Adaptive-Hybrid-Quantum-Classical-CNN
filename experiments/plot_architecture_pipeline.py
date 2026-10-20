import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(16, 9.2), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9.2)
ax.axis('off')

# Professional palette
c_input = '#EBF5FB'
c_backbone = '#D4E6F1'
c_pre = '#FEF9E7'
c_qnode_z = '#E8F8F5'   # Mint green
c_qnode_x = '#FDEDEC'   # Soft coral red
c_gate = '#F4ECF7'      # Soft lavender
c_fuse = '#EAECEE'
c_head = '#F2F4F4'
c_bottom = '#FEFDE8'
c_border = '#1B2631'

# 0. Title & Subtitle
ax.text(8.0, 8.7, 'End-to-End Architecture: Adaptive Dual-QNode Gating Network', 
        ha='center', va='center', fontsize=14.5, fontweight='bold', color=c_border)
ax.text(8.0, 8.3, 'Symmetrical Hybrid Quantum-Classical Framework with Non-Commuting Pauli Readout Routing', 
        ha='center', va='center', fontsize=10.5, style='italic', color='#5D6D7E')

# MAIN FLOW VERTICAL CENTER = 4.8
# 1. Input Image
r1 = patches.FancyBboxPatch((0.4, 3.0), 1.9, 3.6, boxstyle="round,pad=0.1", fc=c_input, ec='#2980B9', lw=1.5)
ax.add_patch(r1)
ax.text(1.35, 5.9, 'OCT B-Scan\n(OCTMNIST)', ha='center', va='center', fontsize=10, fontweight='bold', color='#1B4F72')
ax.text(1.35, 4.3, '28 x 28 x 1\n4 Classes:\nCNV, DME,\nDrusen, Normal', ha='center', va='center', fontsize=8.5, color='#2C3E50', linespacing=1.3)

# Arrow 1 -> 2
ax.annotate('', xy=(2.7, 4.8), xytext=(2.3, 4.8), arrowprops=dict(arrowstyle="->", lw=1.5, color='#2C3E50'))

# 2. ResNet-18 Backbone
r2 = patches.FancyBboxPatch((2.7, 3.0), 2.0, 3.6, boxstyle="round,pad=0.1", fc=c_backbone, ec='#2471A3', lw=1.5)
ax.add_patch(r2)
ax.text(3.7, 5.9, 'Frozen Backbone\nResNet-18', ha='center', va='center', fontsize=10, fontweight='bold', color='#154360')
ax.text(3.7, 4.3, 'ImageNet-1K\nGAP Layer\n512-dim Vector\n(Zero Leakage)', ha='center', va='center', fontsize=8.5, color='#2C3E50', linespacing=1.3)

# Arrow 2 -> 3
ax.annotate('', xy=(5.1, 4.8), xytext=(4.7, 4.8), arrowprops=dict(arrowstyle="->", lw=1.5, color='#2C3E50'))

# 3. Pre-Encoder
r3 = patches.FancyBboxPatch((5.1, 3.0), 1.9, 3.6, boxstyle="round,pad=0.1", fc=c_pre, ec='#D4AC0D', lw=1.5)
ax.add_patch(r3)
ax.text(6.05, 5.9, 'Pre-Encoder', ha='center', va='center', fontsize=10, fontweight='bold', color='#7D6608')
ax.text(6.05, 4.3, 'Linear(512 $\\rightarrow$ 6)\nBatchNorm1d\nReLU Activation\n$x_{\\mathrm{cls}} \\in \\mathbb{R}^6$', ha='center', va='center', fontsize=8.5, color='#5D4037', linespacing=1.3)

# Forking Arrows from Pre-Encoder (3) to 3 Branches
ax.annotate('', xy=(7.4, 6.6), xytext=(7.0, 5.2), arrowprops=dict(arrowstyle="->", lw=1.5, color='#1E8449'))
ax.annotate('', xy=(7.4, 4.8), xytext=(7.0, 4.8), arrowprops=dict(arrowstyle="->", lw=1.5, color='#7D3C98'))
ax.annotate('', xy=(7.4, 3.0), xytext=(7.0, 4.4), arrowprops=dict(arrowstyle="->", lw=1.5, color='#C0392B'))

# Helper for the 3 middle cards to guarantee 100% mathematical symmetry
def draw_card(x, y_center, title, lines, fc, ec, c_title, c_text):
    box = patches.FancyBboxPatch((x, y_center - 0.8), 3.0, 1.6, boxstyle="round,pad=0.1", fc=fc, ec=ec, lw=1.6)
    ax.add_patch(box)
    # Title centered at y_center + 0.38
    ax.text(x + 1.5, y_center + 0.38, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_title)
    # Body centered at y_center - 0.22
    body = "\n".join(lines)
    ax.text(x + 1.5, y_center - 0.22, body, ha='center', va='center', fontsize=8.0, color=c_text, linespacing=1.35)

# 4A. Green Box (QNode-1) - Center Y = 6.6
draw_card(7.4, 6.6, 
          'QNode-1 (Basis $\\sigma_z$)', 
          ['6 Qubits, 1 Layer', 'Rotations: $R_Y(\\theta)$, Entangling: $CZ$', 'Readout: $\\langle \\sigma_z \\rangle \\in [-1, 1]^6$'],
          c_qnode_z, '#27AE60', '#145A32', '#196F3D')

# 4B. Purple Box (Adaptive Gating) - Center Y = 4.8
draw_card(7.4, 4.8,
          'Adaptive Gating Network',
          ['Linear(6 $\\rightarrow$ 32) + ReLU', 'Linear(32 $\\rightarrow$ 2) + Softmax', 'Routing: $w = [w_Z, w_X]^T$'],
          c_gate, '#8E44AD', '#512E5F', '#6C3483')

# 4C. Red Box (QNode-2) - Center Y = 3.0
draw_card(7.4, 3.0, 
          'QNode-2 (Basis $\\sigma_x$)', 
          ['6 Qubits, 1 Layer', 'Rotations: $R_Z(\\phi)$, Entangling: $CNOT$', 'Readout: $\\langle \\sigma_x \\rangle \\in [-1, 1]^6$'],
          c_qnode_x, '#E74C3C', '#922B21', '#A93226')

# Merging Arrows to Fusion Module
ax.annotate('', xy=(10.8, 5.0), xytext=(10.4, 6.6), arrowprops=dict(arrowstyle="->", lw=1.5, color='#1E8449'))
ax.annotate('', xy=(10.8, 4.8), xytext=(10.4, 4.8), arrowprops=dict(arrowstyle="->", lw=1.5, color='#7D3C98'))
ax.annotate('', xy=(10.8, 4.6), xytext=(10.4, 3.0), arrowprops=dict(arrowstyle="->", lw=1.5, color='#C0392B'))

# 5. Adaptive Fusion Module
r5 = patches.FancyBboxPatch((10.8, 3.0), 1.9, 3.6, boxstyle="round,pad=0.1", fc=c_fuse, ec='#7D3C98', lw=1.5)
ax.add_patch(r5)
ax.text(11.75, 5.9, 'Adaptive Fusion', ha='center', va='center', fontsize=10, fontweight='bold', color='#512E5F')
ax.text(11.75, 4.3, '$f_{\\mathrm{fused}} = w_Z f_Z + w_X f_X$\nBatchNorm1d\nFeature: $f \\in \\mathbb{R}^6$', ha='center', va='center', fontsize=8.5, color='#4A235A', linespacing=1.3)

# Arrow 5 -> 6
ax.annotate('', xy=(13.1, 4.8), xytext=(12.7, 4.8), arrowprops=dict(arrowstyle="->", lw=1.5, color='#2C3E50'))

# 6. Regularized Classifier Head
r6 = patches.FancyBboxPatch((13.1, 3.0), 2.5, 3.6, boxstyle="round,pad=0.1", fc=c_head, ec='#34495E', lw=1.5)
ax.add_patch(r6)
ax.text(14.35, 5.9, 'Classifier Head', ha='center', va='center', fontsize=10, fontweight='bold', color='#1B2631')
ax.text(14.35, 4.25, 'FC(6 $\\rightarrow$ 64) + BN + Drop(0.55)\nFC(64 $\\rightarrow$ 32) + BN + Drop(0.45)\nFC(32 $\\rightarrow$ 4) + Softmax\nLoss: Weighted Cross-Entropy', ha='center', va='center', fontsize=7.8, color='#2C3E50', linespacing=1.25)

# Bottom Comparison Box (Symmetrical Twin - Center Y = 1.0)
r_bot = patches.FancyBboxPatch((0.4, 0.4), 15.2, 1.25, boxstyle="round,pad=0.1", fc=c_bottom, ec='#B7950B', lw=1.2)
ax.add_patch(r_bot)
ax.text(8.0, 1.15, 'Symmetrical 1:1 Classical Baseline Twin: Replaces QNode-Z and QNode-X with two Linear(6 $\\rightarrow$ 6) + Tanh layers ($W_{c1}, W_{c2} \\in \\mathbb{R}^{6 \\times 6}$),', 
        ha='center', va='center', fontsize=9.0, fontweight='bold', color='#7D6608')
ax.text(8.0, 0.72, 'sharing identical Pre-Encoder, Gating Network, and Classifier Head, ensuring exact capacity-matching (6,256 Quantum vs. 6,316 Classical Parameters).', 
        ha='center', va='center', fontsize=8.5, color='#7D6608')

plt.tight_layout()

# Save everywhere
paths = [
    "plots/fig_architecture_pipeline.png",
    "plots/fig_architecture_pipeline_fixed.png",
    "paper/figures/fig_architecture_pipeline.png",
    "paper/figures/fig_architecture_pipeline_fixed.png",
    "GD3/manuscript/figures/fig_architecture_pipeline.png",
    "GD3/manuscript/figures/fig_architecture_pipeline_fixed.png"
]
for p in paths:
    plt.savefig(p, bbox_inches='tight')

plt.close()
print("[OK] Saved mathematically identical cards for Green, Purple, and Red branches.")