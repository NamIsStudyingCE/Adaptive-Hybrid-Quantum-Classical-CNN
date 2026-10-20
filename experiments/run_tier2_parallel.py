"""Tier-2 experiment orchestrator: launches all new ablation/noise workers in
parallel, waits, merges seed-partitions, prints a summary.

Workers (low-data regime, 20 epochs, batch 128, canonical 10 seeds):
  1. proposed_noent        (CZ + CNOT removed)          x 10 seeds
  2. proposed_no_cnot      (only CNOT removed)          x 10 seeds
  3. classical_concat      (classical twin of concat)   x 10 seeds
  4/5. proposed_noisy_short   (1e-3/1e-3/1e-2)          x 5 + 5 seeds
  6/7. proposed_noisy_expected (1e-4/1e-4/1e-3)         x 5 + 5 seeds
"""
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
    {"name": "T2-noent",       "models": "proposed_noent",           "seeds": "0,42,100,2023,777,999,1234,5678,1111,2222", "output": "results/results_t2_noent_10seeds.json"},
    {"name": "T2-nocnot",      "models": "proposed_no_cnot",         "seeds": "0,42,100,2023,777,999,1234,5678,1111,2222", "output": "results/results_t2_no_cnot_10seeds.json"},
    {"name": "T2-ccat",        "models": "classical_concat",         "seeds": "0,42,100,2023,777,999,1234,5678,1111,2222", "output": "results/results_t2_classical_concat_10seeds.json"},
    {"name": "T2-noise-short-A",  "models": "proposed_noisy_short",  "seeds": ",".join(map(str, SEEDS_A)), "output": "results/parts/t2_noise_short_A.json"},
    {"name": "T2-noise-short-B",  "models": "proposed_noisy_short",  "seeds": ",".join(map(str, SEEDS_B)), "output": "results/parts/t2_noise_short_B.json"},
    {"name": "T2-noise-exp-A",    "models": "proposed_noisy_expected", "seeds": ",".join(map(str, SEEDS_A)), "output": "results/parts/t2_noise_expected_A.json"},
    {"name": "T2-noise-exp-B",    "models": "proposed_noisy_expected", "seeds": ",".join(map(str, SEEDS_B)), "output": "results/parts/t2_noise_expected_B.json"},
]

N_WORKERS = len(WORKERS)
CPU = os.cpu_count() or 8
THREADS = max(1, CPU // N_WORKERS)

def merge_two_parts(part_a, part_b, output_json, model_name):
    with open(part_a, encoding='utf-8') as f:
        a = json.load(f)
    with open(part_b, encoding='utf-8') as f:
        b = json.load(f)
    runs = a['raw_runs'][model_name] + b['raw_runs'][model_name]
    runs.sort(key=lambda r: ALL_SEEDS.index(r['seed']))
    agg = {}
    for met in METRICS:
        vals = [r['test_metrics'][met] for r in runs]
        agg[met] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'values': [float(v) for v in vals],
        }
    total_runtime = a['metadata'].get('total_runtime_seconds', 0) + b['metadata'].get('total_runtime_seconds', 0)
    out = {
        'metadata': {
            'benchmark_title': 'Tier-2 noise ablation (proposed model under NISQ noise)',
            'dataset': 'OCTMNIST (low_data regime)',
            'regime': 'low_data',
            'num_seeds': len(runs),
            'seeds': [r['seed'] for r in runs],
            'epochs': 20,
            'batch_size': 128,
            'device': 'CUDA/CPU (Multi-Worker Parallel)',
            'total_runtime_seconds': float(total_runtime),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'noise_model': 'depolarizing prep+gates, bit-flip readout',
        },
        'aggregated_metrics': {model_name: agg},
        'raw_runs': {model_name: runs},
    }
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"[OK] merged -> {output_json} ({len(runs)} seeds)")

def main():
    os.makedirs("results/parts", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    print("=" * 80)
    print(f"TIER-2 PARALLEL RUN: {N_WORKERS} workers | CPU={CPU} cores | {THREADS} threads/worker")
    print("=" * 80)

    env = os.environ.copy()
    env['T2_THREADS'] = str(THREADS)
    procs = []
    for cfg in WORKERS:
        cmd = [sys.executable, "-u", "-m", "experiments.benchmark_runner",
               "--regime", "low_data", "--epochs", "20", "--batch-size", "128",
               "--models", cfg["models"], "--seeds", cfg["seeds"],
               "--output-json", cfg["output"]]
        log = open(f"results/logs/{cfg['name']}.log", "w", encoding="utf-8")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        procs.append((cfg["name"], p, log))
        print(f"[*] {cfg['name']:<16} PID={p.pid} models={cfg['models']} seeds={cfg['seeds']}")

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

    print("\n=== MERGE NOISE PARTS ===")
    merge_two_parts("results/parts/t2_noise_short_A.json", "results/parts/t2_noise_short_B.json",
                    "results/results_t2_noise_short_10seeds.json", "proposed_noisy_short")
    merge_two_parts("results/parts/t2_noise_expected_A.json", "results/parts/t2_noise_expected_B.json",
                    "results/results_t2_noise_expected_10seeds.json", "proposed_noisy_expected")

    print("\n=== SUMMARY (accuracy mean +/- std) ===")
    for f in ["results/results_t2_noent_10seeds.json",
              "results/results_t2_no_cnot_10seeds.json",
              "results/results_t2_classical_concat_10seeds.json",
              "results/results_t2_noise_short_10seeds.json",
              "results/results_t2_noise_expected_10seeds.json"]:
        try:
            with open(f, encoding='utf-8') as fh:
                d = json.load(fh)
            for m, metrics in d['aggregated_metrics'].items():
                acc = metrics['accuracy']
                print(f"  {m:<26} {acc['mean']:.4f} +/- {acc['std']:.4f}  (n={len(acc['values'])}) [{os.path.basename(f)}]")
        except Exception as e:
            print(f"  [MISS] {f}: {e}")
    print("\n[DONE] Tier-2 runs complete.")

if __name__ == '__main__':
    main()
