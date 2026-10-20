# Adaptive Gating of Dual-Basis Variational Quantum Circuits: A Rigorous, Symmetrical Benchmark Revealing Structural Degeneracy on OCTMNIST

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35+-brightgreen.svg)](https://pennylane.ai/)
[![Conference](https://img.shields.io/badge/SOICT%202026-Oral%20Presentation-success.svg)](https://soict.org/)

## Overview
Official PyTorch + PennyLane implementation, multi-seed benchmark runner, and statistical validation suite for:

> **Adaptive Gating of Dual-Basis Variational Quantum Circuits: A Rigorous, Symmetrical Benchmark Revealing Structural Degeneracy on OCTMNIST**

This repository provides the complete reproducible code and ground-truth JSON benchmark artifacts evaluating a compact dual-basis Variational Quantum Circuit (VQC) relative to capacity-matched classical baselines for medical image classification — including null results, structural dual-degeneracy verification, and classical control baselines.

---

## Key Results (200 experimental runs across 10 deterministic seeds)

| Finding | Empirical Numbers & Statistical Evaluation |
|---|---|
| **Statistical comparability (low-data)** | Proposed Quantum Gating 68.16% ± 0.0125 vs Classical Twin 68.48% ± 0.0117; paired $t$-test $p=0.4956$, Wilcoxon $p=0.4316$; TOST ±1%: ROC-AUC equivalent ($p_{\text{TOST}}=0.042$), accuracy equivalence not established ($p_{\text{TOST}}=0.083$). |
| **Structural transverse collapse** | Single QNode-$X$: $\langle \sigma_x \rangle$ is identically zero for all inputs due to Pauli-string cancellation $\to$ 25.00% ± 0.0000 by circuit construction (gradients $< 4 \times 10^{-16}$). |
| **Gating gain not significant** | Proposed Gating vs Single QNode-$Z$: +1.17% nominal difference ($p=0.090$ uncorrected, $p=0.452$ Holm-adjusted across six baselines). |
| **Dual degeneracy mechanism** | Branch $\sigma_z$ has $[CZ, \sigma_z] = 0 \implies \langle \sigma_z^{(k)} \rangle = \cos\theta_k \cos x_k$ (six independent single-qubit maps, no entanglement effect). Branch $\sigma_x$ outputs zero. |
| **Classical cosine control (Reviewer 3)** | 6-parameter classical cosine control $f_k(x) = a_k \cos(x_k)$ (5,960 params, matching Single QNode-$Z$ exactly): **0.6764 ± 0.0097** accuracy, ROC-AUC 0.8854 ± 0.0063 ($p=0.2742$ vs Single-Z, $p=0.4034$ vs Proposed Gating). |
| **Data-regime demarcation** | 95.7% accuracy retention at 10% data vs $N=50{,}000$ (classical twin retains 95.4%); at 50k samples, classical twin is nominally higher (71.75% vs 71.20%, not significant after Holm correction). |
| **Capacity parity** | 6,256 (quantum gating) vs 6,316 (classical twin) trainable parameters. |
| **Entanglement-free & classical controls** | `proposed_noent` / `proposed_no_cnot`: 0.6852 ± 0.0138 ($p=0.510$); `classical_concat`: 0.6830 ± 0.0085 ($p=0.752$). |
| **Modeled NISQ noise** | Current level ($10^{-3}$ prep/gate, $10^{-2}$ meas): 0.6825 ± 0.0127 ($p=0.844$); Expected near-term ($10^{-4}$ prep/gate, $10^{-3}$ meas): 0.6800 ± 0.0125 ($p=0.782$) — no significant degradation due to single-qubit dependence. |

Statistical suite: Paired Student's $t$-tests, Wilcoxon signed-rank tests, 95% CIs, Cohen's $d$, Two One-Sided Tests (TOST) of equivalence (±1% pre-specified margin), and Holm–Bonferroni family-wise correction.

---

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate     # or source .venv/bin/activate
pip install -r requirements.txt
```

Data: place the official `octmnist.npz` (MedMNIST v2) in `data/`. The ResNet-18
feature cache is built automatically on first run:

```bash
python src/data.py --cache     # extracts frozen ResNet-18 features (512-d)
```

## Reproduce the benchmark

```bash
# Low-data regime (10% data, N=9,748) — ~75 min on one GPU
python experiments/benchmark_runner.py --regime low_data --seeds 0,42,100,2023,777,999,1234,5678,1111,2222

# Full-data regime (N=50,000) — ~8 h on one GPU
python experiments/benchmark_runner.py --regime full --seeds 0,42,100,2023,777,999,1234,5678,1111,2222

# Reviewer 3: 6-parameter classical cosine control baseline (10 seeds, ~2 min)
python experiments/run_reviewer3_control.py
```

Seeds are fixed and canonical: `{0, 42, 100, 2023, 777, 999, 1234, 5678, 1111, 2222}`.

### Tier-2 ablations & noise study

```bash
# Entanglement ablation + classical-concat control + NISQ noise levels (parallel)
python experiments/run_tier2_parallel.py
```

- `proposed_noent` / `proposed_no_cnot`: entangling gates removed (12 angles kept).
  Note: removing CNOT makes the transverse branch input-dependent again — the
  entangling topology is what silences it (see below).
- `classical_concat`: capacity-matched fully classical control for the
  dual-expectation pipeline.
- `proposed_noisy_short` / `proposed_noisy_expected`: depolarizing prep/gate +
  bit-flip readout noise, executed by a batched torch density-matrix simulator
  (`src/torch_dm_simulator.py`), validated against PennyLane `default.mixed`
  (`experiments/validate_torch_sim.py`, dynamic check:
  `experiments/validate_dynamic.py`).

### Verification utilities

```bash
python experiments/verify_all_consistency.py     # numbers: JSON <-> tables <-> manuscript
python experiments/tost_and_pairwise_tests.py    # TOST + Holm-corrected pairwise tests
python experiments/debug_qnode_x_collapse.py     # proves the structural sigma_x collapse
```

---

## Project Structure

```
Adaptive_Quantum_Gating/
├── data/                       # octmnist.npz + cached ResNet-18 features (gitignored)
├── src/
│   ├── circuits.py             # PennyLane VQCs (entangle flags, noisy builder)
│   ├── torch_dm_simulator.py   # batched torch density-matrix simulator (noise)
│   ├── models.py               # 12 registered architectures (includes CosineBottleneckControl)
│   ├── data.py                 # OCTMNIST loaders & feature caching
│   ├── metrics.py              # 6 metrics + statistical suite
│   └── gating_analysis.py      # routing/entropy plots
├── experiments/
│   ├── trainer.py              # unified training loop
│   ├── benchmark_runner.py     # 10-seed runner
│   ├── run_reviewer3_control.py# Reviewer 3 classical cosine control benchmark
│   ├── run_tier2_parallel.py   # parallel ablation/noise workers
│   ├── validate_torch_sim.py   # static simulator validation
│   ├── validate_dynamic.py     # dynamic (training) simulator validation
│   ├── debug_qnode_x_collapse.py
│   ├── tost_and_pairwise_tests.py
│   ├── plot_architecture_pipeline.py / plot_regime_comparison.py
│   └── verify_all_consistency.py
├── plots/                      # generated benchmark figures
└── results/                    # ground-truth JSON benchmark artifacts
```

## Core Scientific Principles

1. **Anti-overclaim**: null results and structural degeneracies are reported as
   findings, not reframed as wins.
2. **Fair 1:1 baseline** (*Ceteris Paribus*): identical pre-encoder, gating
   network, classifier head; parameter scale matched within <1%.
3. **Multi-seed statistical rigor**: 10 fixed seeds; equivalence testing
   (TOST) and multiplicity correction (Holm) alongside standard tests.
4. **Single source of truth**: every table/figure regenerable from `results/*.json`.
