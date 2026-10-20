"""Complex64 dynamic check: 1 seed x 3 epochs with the simulator only,
compared against the PennyLane reference trajectory recorded by
validate_dynamic.py (PL: val_bacc [0.49771, 0.54251, 0.56901], test 0.54100).
"""
import sys
sys.path.insert(0, '.')
import torch
from src.data import get_octmnist_loaders
from experiments.trainer import train_single_model

SEED = 0
EPOCHS = 3
PL_TRAJ = [0.49771, 0.54251, 0.56901]
PL_TEST = 0.54100

tr, va, te, cw, _, _ = get_octmnist_loaders(batch_size=128, regime='low_data',
                                            seed=SEED, num_classes=4)
r = train_single_model(model_name='proposed_noisy_short', train_loader=tr,
                       val_loader=va, test_loader=te, class_weights=cw,
                       seed=SEED, epochs=EPOCHS, num_classes=4,
                       device='cuda' if torch.cuda.is_available() else 'cpu')
traj = r['history']['val_bacc']
print('sim val_bacc/epoch:', [round(v, 5) for v in traj])
print('PL  val_bacc/epoch:', PL_TRAJ)
print('sim test acc:', round(r['test_metrics']['accuracy'], 5), '| PL test acc:', PL_TEST)
print('sim time:', round(r['training_time_sec'], 1), 's for 3 epochs')
traj_diff = max(abs(x - y) for x, y in zip(traj, PL_TRAJ))
acc_diff = abs(r['test_metrics']['accuracy'] - PL_TEST)
print(f'max |traj diff| = {traj_diff:.5f} | test diff = {acc_diff:.5f}')
ok = traj_diff < 5e-3 and acc_diff < 5e-3
print('COMPLEX64 DYNAMIC CHECK:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
