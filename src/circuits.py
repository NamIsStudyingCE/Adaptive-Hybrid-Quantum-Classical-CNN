import torch
import torch.nn as nn
import pennylane as qml
from pennylane.qnn import TorchLayer

N_QUBITS_DEFAULT = 6
N_LAYERS_DEFAULT = 1

def ansatz_z(weights, n_qubits, entangle=True, p_gate=0.0):
    n_layers = weights.shape[0]
    for l in range(n_layers):
        for i in range(0, n_qubits, 2):
            qml.RY(weights[l, i, 0], wires=i)
            if p_gate > 0:
                qml.DepolarizingChannel(p_gate, wires=i)
        if entangle:
            for i in range(0, n_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
                if p_gate > 0:
                    qml.DepolarizingChannel(p_gate, wires=i)
                    qml.DepolarizingChannel(p_gate, wires=i + 1)
        for i in range(1, n_qubits, 2):
            qml.RY(weights[l, i, 0], wires=i)
            if p_gate > 0:
                qml.DepolarizingChannel(p_gate, wires=i)
        if entangle:
            for i in range(1, n_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
                if p_gate > 0:
                    qml.DepolarizingChannel(p_gate, wires=i)
                    qml.DepolarizingChannel(p_gate, wires=i + 1)
            if n_qubits > 1:
                qml.CZ(wires=[n_qubits - 1, 0])
                if p_gate > 0:
                    qml.DepolarizingChannel(p_gate, wires=n_qubits - 1)
                    qml.DepolarizingChannel(p_gate, wires=0)

def ansatz_x(weights, n_qubits, entangle=True, p_gate=0.0):
    n_layers = weights.shape[0]
    for l in range(n_layers):
        for i in range(0, n_qubits, 2):
            qml.RZ(weights[l, i, 0], wires=i)
            if p_gate > 0:
                qml.DepolarizingChannel(p_gate, wires=i)
        if entangle:
            for i in range(0, n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
                if p_gate > 0:
                    qml.DepolarizingChannel(p_gate, wires=i)
                    qml.DepolarizingChannel(p_gate, wires=i + 1)
        for i in range(1, n_qubits, 2):
            qml.RZ(weights[l, i, 0], wires=i)
            if p_gate > 0:
                qml.DepolarizingChannel(p_gate, wires=i)
        if entangle:
            for i in range(1, n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
                if p_gate > 0:
                    qml.DepolarizingChannel(p_gate, wires=i)
                    qml.DepolarizingChannel(p_gate, wires=i + 1)
            if n_qubits > 1:
                qml.CNOT(wires=[n_qubits - 1, 0])
                if p_gate > 0:
                    qml.DepolarizingChannel(p_gate, wires=n_qubits - 1)
                    qml.DepolarizingChannel(p_gate, wires=0)

def build_dual_qnodes(n_qubits=N_QUBITS_DEFAULT, n_layers=N_LAYERS_DEFAULT, dev_name='default.qubit',
                      entangle_z=True, entangle_x=True):
    dev1 = qml.device(dev_name, wires=n_qubits)
    dev2 = qml.device(dev_name, wires=n_qubits)
    @qml.qnode(dev1, interface='torch', diff_method='backprop')
    def qnode_z(inputs, weights1):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
        ansatz_z(weights1, n_qubits, entangle=entangle_z)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    @qml.qnode(dev2, interface='torch', diff_method='backprop')
    def qnode_x(inputs, weights2):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
        ansatz_x(weights2, n_qubits, entangle=entangle_x)
        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    weight_shapes1 = {'weights1': (n_layers, n_qubits, 1)}
    weight_shapes2 = {'weights2': (n_layers, n_qubits, 1)}
    qnn1 = TorchLayer(qnode_z, weight_shapes1)
    qnn2 = TorchLayer(qnode_x, weight_shapes2)
    return qnn1, qnn2

def build_dual_qnodes_noisy(n_qubits=N_QUBITS_DEFAULT, n_layers=N_LAYERS_DEFAULT, dev_name='default.mixed',
                            p_prep=0.0, p_gate=0.0, p_meas=0.0, diff_method='backprop'):
    """NISQ noise model: depolarizing after state prep, after every gate, and
    bit-flip readout error before each expectation measurement.
    Channels require a mixed-state device (default.mixed)."""
    dev1 = qml.device(dev_name, wires=n_qubits)
    dev2 = qml.device(dev_name, wires=n_qubits)
    @qml.qnode(dev1, interface='torch', diff_method=diff_method)
    def qnode_z(inputs, weights1):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
        if p_prep > 0:
            for i in range(n_qubits):
                qml.DepolarizingChannel(p_prep, wires=i)
        ansatz_z(weights1, n_qubits, entangle=True, p_gate=p_gate)
        if p_meas > 0:
            for i in range(n_qubits):
                qml.BitFlip(p_meas, wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    @qml.qnode(dev2, interface='torch', diff_method=diff_method)
    def qnode_x(inputs, weights2):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
        if p_prep > 0:
            for i in range(n_qubits):
                qml.DepolarizingChannel(p_prep, wires=i)
        ansatz_x(weights2, n_qubits, entangle=True, p_gate=p_gate)
        if p_meas > 0:
            for i in range(n_qubits):
                qml.BitFlip(p_meas, wires=i)
        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    weight_shapes1 = {'weights1': (n_layers, n_qubits, 1)}
    weight_shapes2 = {'weights2': (n_layers, n_qubits, 1)}
    qnn1 = TorchLayer(qnode_z, weight_shapes1)
    qnn2 = TorchLayer(qnode_x, weight_shapes2)
    return qnn1, qnn2

def build_single_qnode(observable='Z', n_qubits=N_QUBITS_DEFAULT, n_layers=N_LAYERS_DEFAULT, dev_name='default.qubit'):
    dev = qml.device(dev_name, wires=n_qubits)
    if observable == 'Z':
        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
            ansatz_z(weights, n_qubits)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    elif observable == 'X':
        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
            ansatz_x(weights, n_qubits)
            return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    else:
        raise ValueError('Unknown observable: ' + observable)
    weight_shapes = {'weights': (n_layers, n_qubits, 1)}
    return TorchLayer(qnode, weight_shapes)

if __name__ == '__main__':
    print('Testing Dual QNodes...')
    qnn1, qnn2 = build_dual_qnodes(n_qubits=6, n_layers=1)
    x = torch.randn(4, 6)
    out1 = qnn1(x)
    out2 = qnn2(x)
    print('QNN1 shape:', out1.shape, 'QNN2 shape:', out2.shape)
