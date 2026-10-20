import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import json

LOG_PATH = r"C:\Users\nghna\.gemini\antigravity\brain\5caca392-c5ac-4fed-a40a-b56bde601801\.system_generated\tasks\task-941.log"
PARTS_DIR = "results/parts"
OUTPUT_PART0 = os.path.join(PARTS_DIR, "full_part_0.json")

MODELS = [
    "proposed_dual_qnode",
    "symmetrical_classical",
    "dual_qnode_concat",
    "dual_qnode_fixed_sum",
    "single_qnode_z",
    "single_qnode_x",
    "pure_classical"
]

def parse_log_for_seeds(log_path=LOG_PATH, target_seeds=[0, 42, 100], output_json=OUTPUT_PART0):
    os.makedirs(PARTS_DIR, exist_ok=True)
    if not os.path.exists(log_path):
        print(f"[!] Log file not found: {log_path}")
        return False
        
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        
    raw_runs = {m: [] for m in MODELS}
    current_model = None
    current_seed = None
    current_epochs = 20
    
    # Pattern 1: [1/70] Running proposed_dual_qnode       | Seed: 0 | Epochs: 20 | Dev: CUDA
    run_pattern = re.compile(r"\[\d+/70\] Running\s+(\w+)\s+\|\s+Seed:\s+(\d+)\s+\|\s+Epochs:\s+(\d+)")
    # Pattern 2: -> Acc: 0.7020 | B-Acc: 0.7020 | F1: 0.6925 | MCC: 0.6064 | ROC-AUC: 0.9024 | PR-AUC: 0.7716 (501.7s)
    metric_pattern = re.compile(r"-> Acc:\s+([\d\.]+)\s+\|\s+B-Acc:\s+([\d\.]+)\s+\|\s+F1:\s+([\d\.]+)\s+\|\s+MCC:\s+([\d\.]+)\s+\|\s+ROC-AUC:\s+([\d\.]+)\s+\|\s+PR-AUC:\s+([\d\.]+)\s+\(([\d\.]+)s\)")
    
    for line in lines:
        m_run = run_pattern.search(line)
        if m_run:
            m_name, s_str, ep_str = m_run.groups()
            current_model = m_name
            current_seed = int(s_str)
            current_epochs = int(ep_str)
            continue
            
        m_met = metric_pattern.search(line)
        if m_met and current_model and current_seed in target_seeds:
            acc, bacc, f1, mcc, roc, pr, t_sec = m_met.groups()
            run_dict = {
                "model": current_model,
                "seed": current_seed,
                "epochs": current_epochs,
                "training_time_sec": float(t_sec),
                "test_metrics": {
                    "accuracy": float(acc),
                    "balanced_accuracy": float(bacc),
                    "f1_macro": float(f1),
                    "mcc": float(mcc),
                    "roc_auc": float(roc),
                    "pr_auc": float(pr)
                }
            }
            # Check if not already added
            if not any(r["seed"] == current_seed for r in raw_runs[current_model]):
                raw_runs[current_model].append(run_dict)
                
    total_parsed = sum(len(raw_runs[m]) for m in MODELS)
    print(f"[*] Parsed {total_parsed} runs across target seeds {target_seeds}:")
    for m in MODELS:
        print(f"    - {m:<25}: {len(raw_runs[m])} runs")
        
    if total_parsed == len(MODELS) * len(target_seeds):
        part_data = {
            "metadata": {
                "regime": "full",
                "seeds": target_seeds,
                "epochs": 20,
                "batch_size": 128,
                "num_runs": total_parsed
            },
            "raw_runs": raw_runs
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(part_data, f, indent=2)
        print(f"[OK] Successfully saved Part 0 to: {output_json}")
        return True
    else:
        print(f"[!] Target runs not yet fully complete ({total_parsed}/{len(MODELS)*len(target_seeds)})")
        return False

if __name__ == "__main__":
    parse_log_for_seeds()
