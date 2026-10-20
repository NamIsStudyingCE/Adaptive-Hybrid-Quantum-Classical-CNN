# Arbitration: teacher's simulation vs the actual project circuit.
import sys
sys.path.insert(0, '.')
import numpy as np
import torch
import pennylane as qml

n = 6
dev = qml.device('default.qubit', wires=n)
rng = np.random.default_rng(7)

def x_expectations(inputs, phi, mode):
    """mode 'paper': interleaved (RZ even -> CNOT even -> RZ odd -> CNOT odd -> ring), as in src/circuits.py
       mode 'naive': single RZ layer on all wires, then the ring (likely the teacher's reimplementation)"""
    @qml.qnode(dev, interface='autograd', diff_method='best')
    def q():
        qml.AngleEmbedding(inputs, wires=range(n), rotation='X')
        if mode == 'paper':
            for i in range(0, n, 2):
                qml.RZ(phi[i], wires=i)
            for i in range(0, n - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(1, n, 2):
                qml.RZ(phi[i], wires=i)
            for i in range(1, n - 1, 2):
                qml.CNOT(wires=[i, i + 1])
        else:
            for i in range(n):
                qml.RZ(phi[i], wires=i)
            for i in range(0, n - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(1, n - 1, 2):
                qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n - 1, 0])
        return [qml.expval(qml.PauliX(i)) for i in range(n)]
    return np.array(q())

print('=== 1. TorchLayer init: phi cua QNode-X trong project ===')
from src.circuits import build_single_qnode
qnn = build_single_qnode('X', n_qubits=6, n_layers=1)
w0 = qnn.weights.detach().numpy().flatten()
print('  init weights (1 layer, 6 angles):', np.round(w0, 4))
print('  all zero?', bool(np.allclose(w0, 0)), '| min =', round(w0.min(), 4), '| max =', round(w0.max(), 4))

print()
print('=== 2. Spread cua <sigma_x> tren MACH THAT (interleaved) ===')
xs = rng.normal(0, 1, (500, n))
for label, phi in [('default init (TorchLayer)', torch.tensor(w0).float()),
                   ('phi = 0 (zero-init)', np.zeros(n)),
                   ('phi ~ U(0, 2pi)', rng.uniform(0, 2 * np.pi, n)),
                   ('phi ~ N(0, 1)', rng.normal(0, 1, n))]:
    ph = phi.numpy() if hasattr(phi, 'numpy') else phi
    outs = np.array([x_expectations(x, ph, 'paper') for x in xs])
    print(f'  {label:26} spread = {outs.max(0) - outs.min(0)}')

print()
print('=== 3. Ban NAIVE (RZ toan bo wire -> CNOT ring) — kha nang cao la ban thay mo phong ===')
for label, phi in [('phi ~ N(0, 1)', rng.normal(0, 1, n)), ('phi = 0', np.zeros(n))]:
    outs = np.array([x_expectations(x, phi, 'naive') for x in xs])
    print(f'  {label:26} spread = {outs.max(0) - outs.min(0)}')
