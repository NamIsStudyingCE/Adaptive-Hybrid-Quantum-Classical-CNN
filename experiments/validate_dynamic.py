"""Dynamic validation: short training run (1 seed x 3 epochs) on both backends.
Simulator passes only if the training trajectory matches the PennyLane
reference within float-chaos tolerance (|diff| < 5e-3 on val B-Acc per epoch
and test accuracy).
"""
import sys
sys.path.insert(0, '.')
import torch
from src.data import get_octmnist_loaders
from experiments.trainer import train_single_model

SEED = 0
EPOCHS = 3
tr, va, te, cw, _, _ = get_octmnist_loaders(batch_size=128, regime='low_data',
                                            seed=SEED, num_classes=4)

res = {}
for name in ['proposed_noisy_short', 'proposed_noisy_pl']:
    r = train_single_model(model_name=name, train_loader=tr, val_loader=va,
                           test_loader=te, class_weights=cw, seed=SEED,
                           epochs=EPOCHS, num_classes=4,
                           device='cuda' if torch.cuda.is_available() else 'cpu')
    res[name] = r
    print(f"[{name}] val_bacc/epoch: {[round(v, 5) for v in r['history']['val_bacc']]}")
    print(f"[{name}] test acc: {r['test_metrics']['accuracy']:.5f} "
          f"roc: {r['test_metrics']['roc_auc']:.5f} time: {r['training_time_sec']:.1f}s")

a, b = res['proposed_noisy_short'], res['proposed_noisy_pl']
traj_diff = max(abs(x - y) for x, y in zip(a['history']['val_bacc'], b['history']['val_bacc']))
acc_diff = abs(a['test_metrics']['accuracy'] - b['test_metrics']['accuracy'])
print(f"\nmax |val_bacc diff| across epochs : {traj_diff:.5f}")
print(f"|test acc diff|                   : {acc_diff:.5f}")
ok = traj_diff < 5e-3 and acc_diff < 5e-3
print('DYNAMIC VALIDATION:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
