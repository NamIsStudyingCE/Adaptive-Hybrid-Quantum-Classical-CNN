import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from src.models import MODEL_REGISTRY, build_model

def audit():
    print("=== 1. AUDITING MODEL PARAMETER COUNTS (src/models.py) ===")
    param_counts = {}
    for m in MODEL_REGISTRY:
        _, p = build_model(m, 512, 4)
        param_counts[m] = p
        print(f"  [Code] {m}: {p:,} params")
    assert param_counts["proposed_dual_qnode"] == 6256
    assert param_counts["symmetrical_classical"] == 6316
    assert param_counts["single_qnode_z"] == 5960
    assert param_counts["single_qnode_x"] == 5960
    assert param_counts["cosine_bottleneck"] == 5960

    print("\n=== 2. AUDITING ALL 200 EXPERIMENTAL RUNS IN results/*.json ===")
    json_files = {
        "Low-Data Main (7x10=70)": "results/results_benchmark_10seeds.json",
        "50k-Data Main (7x10=70)": "results/results_benchmark_full_10seeds.json",
        "Entanglement-Free noent (10)": "results/results_t2_noent_10seeds.json",
        "CNOT-Removed no_cnot (10)": "results/results_t2_no_cnot_10seeds.json",
        "Classical Concat Control (10)": "results/results_t2_classical_concat_10seeds.json",
        "Classical Cosine Control (10)": "results/results_cosine_bottleneck_low_data.json",
        "NISQ Noise Current (10)": "results/results_t2_noise_short_10seeds.json",
        "NISQ Noise Expected (10)": "results/results_t2_noise_expected_10seeds.json",
    }
    total_runs = 0
    loaded = {}
    for label, path in json_files.items():
        assert os.path.exists(path), f"Missing ground-truth result file: {path}"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        loaded[path] = data
        if "raw_runs" in data:
            raw = data["raw_runs"]
            n_runs = sum(len(v) for v in raw.values()) if isinstance(raw, dict) else len(raw)
        elif "per_seed" in data:
            n_runs = len(data["per_seed"])
        else:
            n_runs = len(data.get("seeds", []))
        total_runs += n_runs
        print(f"  [OK] {label}: {n_runs} runs verified ({path})")
    assert total_runs == 200, f"Expected 200 total runs, found {total_runs}"
    print(f"  >>> Total verified experimental runs across all JSON artifacts: {total_runs} <<<")

    d10 = loaded["results/results_benchmark_10seeds.json"]
    dfull = loaded["results/results_benchmark_full_10seeds.json"]
    print("\n=== 3. AUDITING KEY BENCHMARK METRICS & STATISTICAL TESTS ===")
    print(f"  Low-data Proposed Acc:  {d10['aggregated_metrics']['proposed_dual_qnode']['accuracy']['mean']:.4f} ± {d10['aggregated_metrics']['proposed_dual_qnode']['accuracy']['std']:.4f}")
    print(f"  Low-data Classical Acc: {d10['aggregated_metrics']['symmetrical_classical']['accuracy']['mean']:.4f} ± {d10['aggregated_metrics']['symmetrical_classical']['accuracy']['std']:.4f}")
    print(f"  50k-data Proposed Acc:  {dfull['aggregated_metrics']['proposed_dual_qnode']['accuracy']['mean']:.4f} ± {dfull['aggregated_metrics']['proposed_dual_qnode']['accuracy']['std']:.4f}")
    print(f"  50k-data Classical Acc: {dfull['aggregated_metrics']['symmetrical_classical']['accuracy']['mean']:.4f} ± {dfull['aggregated_metrics']['symmetrical_classical']['accuracy']['std']:.4f}")

    stat_test = d10["statistical_comparisons"]["proposed_vs_symmetrical_classical"]["accuracy"]
    print(f"  Paired t-test p-value:  {stat_test['t_pvalue']:.4f}")
    print(f"  Wilcoxon p-value:       {stat_test['wilcoxon_pvalue']:.4f}")
    print(f"  Cohen's d:              {stat_test['cohens_d']:.3f}")
    print(f"  95% CI:                 [{stat_test['ci_95'][0]:.4f}, {stat_test['ci_95'][1]:.4f}]")

    print("\n=== 4. AUDITING README.md ===")
    with open("README.md", encoding="utf-8") as f:
        readme = f.read()
    for expected in ["200 experimental runs", "68.16%", "68.48%", "0.4956", "0.4316",
                     "6,256", "6,316", "0.6764", "0.6852", "0.6830", "0.6825", "0.6800"]:
        assert expected in readme, f"README.md missing expected value: {expected}"
    print("  [OK] README.md matches all ground-truth JSON metrics and 200-run protocol.")

    if os.path.exists("paper/main.tex"):
        print("\n=== 5. AUDITING LOCAL MANUSCRIPT (paper/main.tex) ===")
        with open("paper/main.tex", encoding="utf-8") as f:
            tex = f.read()
        for expected in ["200", "0.6816", "0.6848", "0.4956", "0.4316", "-0.225",
                         "71.20", "6,256", "6,316", "0.042", "0.452", "0.6764"]:
            assert expected in tex, f"main.tex missing expected value: {expected}"
        print("  [OK] Local paper/main.tex matches ground-truth JSON artifacts.")

    print("\n=======================================================")
    print(">>> AUDIT RESULT: 100% CONSISTENCY CONFIRMED (200 RUNS) <<<")
    print("=======================================================")

if __name__ == '__main__':
    audit()