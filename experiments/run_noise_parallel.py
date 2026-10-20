"""NISQ-noise benchmark orchestrator: 4 parallel workers (2 noise levels x
2 seed-halves), then merges parts into 10-seed ground-truth JSONs."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import subprocess
import time
import numpy as np

ALL_SEEDS = [0, 42, 100, 2023, 777, 999, 1234, 5678, 1111, 2222]
SEEDS_A = ALL_SEEDS[:5]
SEEDS_B = ALL_SEEDS[5:]
METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "mcc", "roc_auc", "pr_auc"]

WORKERS = [
    {"name": "N-short-A", "models": "proposed_noisy_short",    "seeds": ",".join(map(str, SEEDS_A)), "output": "results/parts/t2_noise_short_A.json"},
    {"name": "N-short-B", "models": "proposed_noisy_short",    "seeds": ",".join(map(str, SEEDS_B)), "output": "results/parts/t2_noise_short_B.json"},
    {"name": "N-exp-A",   "models": "proposed_noisy_expected", "seeds": ",".join(map(str, SEEDS_A)), "output": "results/parts/t2_noise_expected_A.json"},
    {"name": "N-exp-B",   "models": "proposed_noisy_expected", "seeds": ",".join(map(str, SEEDS_B)), "output": "results/parts/t2_noise_expected_B.json"},
]

def merge_two_parts(part_a, part_b, output_json, model_name):
    a = json.load(open(part_a, encoding='utf-8'))
    b = json.load(open(part_b, encoding='utf-8'))
    runs = a['raw_runs'][model_name] + b['raw_runs'][model_name]
    runs.sort(key=lambda r: ALL_SEEDS.index(r['seed']))
    agg = {}
    for met in METRICS:
        vals = [r['test_metrics'][met] for r in runs]
        agg[met] = {'mean': float(np.mean(vals)),
                    'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    'min': float(np.min(vals)), 'max': float(np.max(vals)),
                    'values': [float(v) for v in vals]}
    total_runtime = a['metadata'].get('total_runtime_seconds', 0) + b['metadata'].get('total_runtime_seconds', 0)
    out = {
        'metadata': {
            'benchmark_title': 'Tier-2 noise ablation (proposed model under NISQ noise)',
            'dataset': 'OCTMNIST (low_data regime)',
            'regime': 'low_data', 'num_seeds': len(runs),
            'seeds': [r['seed'] for r in runs],
            'epochs': 20, 'batch_size': 128,
            'device': 'CUDA (torch-dm simulator, complex64, validated)',
            'total_runtime_seconds': float(total_runtime),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'noise_model': 'depolarizing prep+gates, bit-flip readout',
        },
        'aggregated_metrics': {model_name: agg},
        'raw_runs': {model_name: runs},
    }
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"[OK] merged -> {output_json} ({len(runs)} seeds)")

def main():
    os.makedirs("results/parts", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    cpu = os.cpu_count() or 8
    env = os.environ.copy()
    env['T2_THREADS'] = str(max(1, cpu // len(WORKERS)))
    print(f"NOISE RUN: {len(WORKERS)} workers | CPU={cpu}")
    procs = []
    for cfg in WORKERS:
        cmd = [sys.executable, "-u", "-m", "experiments.benchmark_runner",
               "--regime", "low_data", "--epochs", "20", "--batch-size", "128",
               "--models", cfg["models"], "--seeds", cfg["seeds"],
               "--output-json", cfg["output"]]
        log = open(f"results/logs/{cfg['name']}.log", "w", encoding="utf-8")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        procs.append((cfg["name"], p, log))
        print(f"[*] {cfg['name']:<12} PID={p.pid} seeds={cfg['seeds']}")
    t0 = time.time()
    while True:
        done = all(p.poll() is not None for _, p, _ in procs)
        status = " | ".join(f"{n}:{'RUN' if p.poll() is None else f'END({p.poll()})'}" for n, p, _ in procs)
        print(f"[{int((time.time()-t0)/60):>3}min] {status}", flush=True)
        if done:
            break
        time.sleep(60)
    for _, _, log in procs:
        log.close()
    print("=== MERGE ===")
    merge_two_parts("results/parts/t2_noise_short_A.json", "results/parts/t2_noise_short_B.json",
                    "results/results_t2_noise_short_10seeds.json", "proposed_noisy_short")
    merge_two_parts("results/parts/t2_noise_expected_A.json", "results/parts/t2_noise_expected_B.json",
                    "results/results_t2_noise_expected_10seeds.json", "proposed_noisy_expected")
    print("=== SUMMARY (accuracy) ===")
    noiseless = 0.6816
    for f, m in [("results/results_t2_noise_short_10seeds.json", "proposed_noisy_short"),
                 ("results/results_t2_noise_expected_10seeds.json", "proposed_noisy_expected")]:
        d = json.load(open(f, encoding='utf-8'))
        a = d['aggregated_metrics'][m]['accuracy']
        print(f"  {m:<26} {a['mean']:.4f} +/- {a['std']:.4f}  (noiseless ref: {noiseless})")
    print("[DONE]")

if __name__ == '__main__':
    main()
