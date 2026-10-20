"""Debug: Is Single QNode-X collapse structural (input-independent <sigma_x>)?
Teacher's hypothesis: with R_X AngleEmbedding followed by RZ rotations, <sigma_x>
is independent of the input -> collapse to 25.00% is a circuit artifact, not a
learned phenomenon. This script tests input-dependence of every readout wire.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import torch
from src.circuits import build_single_qnode

torch.manual_seed(0)
np.random.seed(0)

print('=' * 78)
print('TEST 1: Input-dependence of readout features (fixed random weights)')
print('=' * 78)

# Simulate RobustScaler-scaled inputs: roughly N(0,1), some outliers like real data
for obs, name in [('Z', 'QNode-Z (RY/CZ ansatz)   '), ('X', 'QNode-X (RZ/CNOT ansatz)   ')]:
    qnn = build_single_qnode(observable=obs, n_qubits=6, n_layers=1)
    qnn.eval()
    with torch.no_grad():
        # 500 diverse inputs x 5 random weight draws
        spread_per_wire = np.zeros(6)
        for trial in range(5):
            xs = torch.randn(500, 6)
            outs = qnn(xs).numpy()          # (500, 6) expvals per wire
            spread_per_wire = np.maximum(spread_per_wire, outs.max(axis=0) - outs.min(axis=0))
    print(f'{name} | max spread (max-min over 500 inputs) per wire: {np.round(spread_per_wire, 8)}')

print()
print('=' * 78)
print('TEST 2: d<sigma_x>/d(input) gradient magnitude (structural check)')
print('=' * 78)
for obs, name in [('Z', 'QNode-Z'), ('X', 'QNode-X')]:
    qnn = build_single_qnode(observable=obs, n_qubits=6, n_layers=1)
    xs = torch.randn(64, 6, requires_grad=True)
    outs = qnn(xs)
    outs.sum().backward()
    g = xs.grad.abs().max().item()
    print(f'{name}: max |d(expval)/d(input)| over 64 samples x 6 wires = {g:.3e}')

print()
print('=' * 78)
print('TEST 3: Mechanism decomposition for QNode-X (which piece kills input info?)')
print('=' * 78)
import pennylane as qml
n = 6
dev = qml.device('default.qubit', wires=n)
rng = np.random.default_rng(1)
x = rng.normal(0, 1, n)
w = rng.uniform(0, np.pi, (n, 1))

def make_circuit(use_entangle=True, rz_instead_of_ry=False, embed='X'):
    @qml.qnode(dev, interface='autograd', diff_method='parameter-shift')
    def q(inp, wt):
        if embed == 'X':
            qml.AngleEmbedding(inp, wires=range(n), rotation='X')
        else:
            qml.AngleEmbedding(inp, wires=range(n), rotation='Z')
        for i in range(n):
            if rz_instead_of_ry:
                qml.RZ(wt[i, 0], wires=i)
            else:
                qml.RY(wt[i, 0], wires=i)
        if use_entangle:
            for i in range(0, n - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(1, n - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n - 1, 0])
        return [qml.expval(qml.PauliX(i)) for i in range(n)]
    return q

variants = [
    ('(a) Full paper circuit: RX-embed + RZ-ansatz + CNOT-ring', dict(use_entangle=True,  rz_instead_of_ry=True,  embed='X')),
    ('(b) No entanglement:     RX-embed + RZ-ansatz only     ', dict(use_entangle=False, rz_instead_of_ry=True,  embed='X')),
    ('(c) RY instead of RZ:     RX-embed + RY-ansatz + CNOT  ', dict(use_entangle=True,  rz_instead_of_ry=False, embed='X')),
    ('(d) Z-embed control:      RZ-embed + RZ-ansatz + CNOT  ', dict(use_entangle=True,  rz_instead_of_ry=True,  embed='Z')),
]
for label, kw in variants:
    q = make_circuit(**kw)
    outs = np.array([q(x, w), q(rng.normal(0, 1, n), w),
                     q(rng.normal(0, 5, n), w)])
    spread = outs.max(axis=0) - outs.min(axis=0)
    print(f'{label} spread per wire: {np.round(spread, 6)}')

print()
print('VERDICT: if (a) spread == 0 for all wires -> <sigma_x> is provably')
print('input-independent: collapse is a structural artifact of the circuit choice.')
