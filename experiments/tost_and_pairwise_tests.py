"""Teacher's Tier-1 statistics (code-only, no retraining):
V2: TOST equivalence test (+-1% absolute accuracy margin) on existing JSON.
V4: Pairwise tests for ALL pairs vs proposed (esp. gating vs single-Z +1.17%),
    with Holm-Bonferroni correction across 6 comparisons.
"""
import sys
sys.path.insert(0, '.')
import json
import numpy as np
from scipy import stats

with open('results/results_benchmark_10seeds.json') as f:
    d = json.load(f)

runs = d['raw_runs']
MODELS = ['proposed_dual_qnode', 'symmetrical_classical', 'dual_qnode_concat',
          'dual_qnode_fixed_sum', 'single_qnode_z', 'single_qnode_x', 'pure_classical']
LABEL = {'proposed_dual_qnode': 'Proposed Gating', 'symmetrical_classical': 'Classical Twin',
         'dual_qnode_concat': 'Concat', 'dual_qnode_fixed_sum': 'Fixed Sum',
         'single_qnode_z': 'Single-Z', 'single_qnode_x': 'Single-X', 'pure_classical': 'Pure Classical'}

def get_vals(model, metric='accuracy'):
    return np.array([r['test_metrics'][metric] for r in runs[model]])

def paired_tests(a, b):
    diff = a - b
    n = len(diff)
    t, p_two = stats.ttest_rel(a, b)
    p_low = stats.ttest_1samp(diff, -0.01, alternative='greater').pvalue   # H0: diff<=-1%
    p_high = stats.ttest_1samp(diff, +0.01, alternative='less').pvalue     # H0: diff>=+1%
    p_tost = max(p_low, p_high)
    w = stats.wilcoxon(a, b)
    return dict(mean_diff=float(diff.mean()), t=float(t), p_two=float(p_two),
                p_tost=float(p_tost), W=float(w.statistic), p_w=float(w.pvalue))

print('=' * 90)
print('V2: TOST EQUIVALENCE (+-1% abs accuracy margin) — proposed vs each baseline')
print('   TOST passes if p_tost < 0.05 (equivalence established at margin 1%)')
print('=' * 90)
prop = get_vals('proposed_dual_qnode')
for m in MODELS[1:]:
    r = paired_tests(prop, get_vals(m))
    verdict = 'EQUIVALENT (TOST pass)' if r['p_tost'] < 0.05 else 'equivalence NOT established'
    print(f"{LABEL[m]:<16} d={r['mean_diff']:+.4f}  t={r['t']:+.3f}  p_two={r['p_two']:.4f}  "
          f"p_TOST={r['p_tost']:.4f}  -> {verdict}")

print()
print('=' * 90)
print('V4: PAIRWISE TESTS, proposed vs every other model (10 seeds, accuracy)')
print('   Holm-Bonferroni over the m comparisons below')
print('=' * 90)
pairs = []
for m in MODELS[1:]:
    r = paired_tests(prop, get_vals(m))
    pairs.append((m, r))
    print(f"{LABEL[m]:<16} d={r['mean_diff']:+.4f}  t={r['t']:+.3f}  p={r['p_two']:.4f}  "
          f"W={r['W']:.1f}  p_w={r['p_w']:.4f}")

# Holm correction on the two-sided p-values
raw = np.array([p[1]['p_two'] for p in pairs])
order = np.argsort(raw)
holm = np.empty_like(raw)
m = len(raw)
running = 0.0
for rank, idx in enumerate(order):
    val = (m - rank) * raw[idx]
    running = max(running, val)
    holm[idx] = min(1.0, running)
print()
print('Holm-adjusted p-values (alpha=0.05):')
for (mname, r), hp in zip(pairs, holm):
    sig = 'SIGNIFICANT' if hp < 0.05 else 'ns'
    print(f"  proposed vs {LABEL[mname]:<16} p_holm={hp:.4f}  ({sig})")

print()
print('=' * 90)
print('F1-macro and ROC-AUC for the gating-vs-single-Z claim (+1.17% headline)')
print('=' * 90)
for met in ['accuracy', 'f1_macro', 'roc_auc']:
    a, b = get_vals('proposed_dual_qnode', met), get_vals('single_qnode_z', met)
    r = paired_tests(a, b)
    print(f"gating vs single-Z [{met:<9}] d={r['mean_diff']:+.4f}  t={r['t']:+.3f}  p={r['p_two']:.4f}  p_w={r['p_w']:.4f}")

# Also: classical twin vs proposed across ALL 6 metrics for the revised Table 4 note
print()
print('ALL 6 metrics, proposed vs classical twin (two-sided p + TOST):')
for met in ['accuracy', 'balanced_accuracy', 'f1_macro', 'mcc', 'roc_auc', 'pr_auc']:
    r = paired_tests(get_vals('proposed_dual_qnode', met), get_vals('symmetrical_classical', met))
    print(f"  {met:<18} d={r['mean_diff']:+.4f}  p={r['p_two']:.4f}  p_TOST={r['p_tost']:.4f}")
