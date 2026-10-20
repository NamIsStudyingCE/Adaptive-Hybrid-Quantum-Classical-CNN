import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess
import time

WORKER_CONFIGS = [
    {"name": "Worker-1", "seeds": "2023,777", "output": "results/parts/full_part_1.json"},
    {"name": "Worker-2", "seeds": "999,1234", "output": "results/parts/full_part_2.json"},
    {"name": "Worker-3", "seeds": "5678,1111,2222", "output": "results/parts/full_part_3.json"}
]

def launch_workers():
    os.makedirs("results/parts", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    
    processes = []
    print("=" * 80)
    print("LAUNCHING MULTI-WORKER PARALLEL BENCHMARK (SEEDS: 2023, 777, 999, 1234, 5678, 1111, 2222)")
    print("=" * 80)
    
    for cfg in WORKER_CONFIGS:
        cmd = [
            sys.executable, "-u", "-m", "experiments.benchmark_runner",
            "--regime", "full",
            "--epochs", "20",
            "--batch-size", "128",
            "--seeds", cfg["seeds"],
            "--output-json", cfg["output"]
        ]
        log_file = open(f"results/logs/{cfg['name']}.log", "w", encoding="utf-8")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((cfg["name"], p, log_file))
        print(f"[*] Started {cfg['name']} (PID: {p.pid}) -> Seeds: {cfg['seeds']} -> Output: {cfg['output']}")
        
    print("\n[*] All 3 workers launched in parallel! Monitoring progress...")
    
    while True:
        all_done = True
        statuses = []
        for name, p, log_f in processes:
            ret = p.poll()
            if ret is None:
                all_done = False
                statuses.append(f"{name}: RUNNING")
            else:
                statuses.append(f"{name}: DONE (code {ret})")
        print(f"[{time.strftime('%H:%M:%S')}] Status: " + " | ".join(statuses))
        if all_done:
            break
        time.sleep(30)
        
    for name, p, log_f in processes:
        log_f.close()
        
    print("\n[OK] All parallel workers finished! Proceeding to merge partitions...")
    from experiments.merge_benchmark_parts import merge_partitions
    merge_partitions()

if __name__ == "__main__":
    launch_workers()
