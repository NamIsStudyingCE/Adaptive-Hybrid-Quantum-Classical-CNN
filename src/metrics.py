import numpy as np, scipy.stats as stats, torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix)

def compute_all_metrics(y_true, y_probs, num_classes=4):
    if isinstance(y_true, torch.Tensor): y_true = y_true.cpu().numpy()
    if isinstance(y_probs, torch.Tensor): y_probs = y_probs.cpu().numpy()
    y_true = y_true.astype(int)
    y_pred = np.argmax(y_probs, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    b_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    try:
        if num_classes == 2: roc_auc = float(roc_auc_score(y_true, y_probs[:, 1]))
        else: roc_auc = float(roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro'))
    except Exception: roc_auc = float('nan')
    try:
        if num_classes == 2: pr_auc = float(average_precision_score(y_true, y_probs[:, 1]))
        else:
            pr_scores = [average_precision_score((y_true == c).astype(int), y_probs[:, c]) for c in range(num_classes) if np.sum(y_true == c) > 0]
            pr_auc = float(np.mean(pr_scores)) if len(pr_scores) > 0 else float('nan')
    except Exception: pr_auc = float('nan')
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return {"accuracy": acc, "balanced_accuracy": b_acc, "f1_macro": f1_macro, "mcc": mcc, "roc_auc": roc_auc, "pr_auc": pr_auc}, cm

def compute_statistical_comparison(scores_proposed, scores_baseline):
    prop = np.asarray(scores_proposed, dtype=float)
    base = np.asarray(scores_baseline, dtype=float)
    diff = prop - base
    n = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se_diff = std_diff / np.sqrt(n) if n > 0 else 0.0
    if n > 1 and se_diff > 0:
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_low = float(mean_diff - t_crit * se_diff)
        ci_high = float(mean_diff + t_crit * se_diff)
    else: ci_low, ci_high = mean_diff, mean_diff
    if n > 1 and np.any(diff != 0):
        t_stat, t_pval = stats.ttest_rel(prop, base)
        t_stat, t_pval = float(t_stat), float(t_pval)
    else: t_stat, t_pval = 0.0, 1.0
    if n >= 5 and np.any(diff != 0):
        try:
            w_res = stats.wilcoxon(prop, base, alternative='two-sided')
            w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
        except Exception: w_stat, w_pval = float('nan'), 1.0
    else: w_stat, w_pval = float('nan'), 1.0
    cohens_d = float(mean_diff / std_diff) if std_diff > 1e-9 else 0.0
    return {"mean_diff": mean_diff, "std_diff": std_diff, "ci_95": [ci_low, ci_high], "t_stat": t_stat, "t_pvalue": t_pval, "wilcoxon_stat": w_stat, "wilcoxon_pvalue": w_pval, "cohens_d": cohens_d}

def format_significance_stars(p_value):
    if p_value < 0.001: return '***'
    elif p_value < 0.01: return '**'
    elif p_value < 0.05: return '*'
    else: return ' ns'
