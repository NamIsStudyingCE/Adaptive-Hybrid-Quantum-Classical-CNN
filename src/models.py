import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from src.circuits import build_dual_qnodes, build_single_qnode, build_dual_qnodes_noisy
from src.torch_dm_simulator import TorchNoisyDualQNode

class PreEncoder(nn.Module):
    def __init__(self, in_features=512, out_features=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class ClassifierHead(nn.Module):
    def __init__(self, in_features=6, num_classes=4, drop1=0.55, drop2=0.45):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(drop1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(drop2),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class GatingNetwork(nn.Module):
    def __init__(self, in_features=6, hidden_dim=32, num_experts=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
    def forward(self, x):
        logits = self.mlp(x)
        weights = torch.softmax(logits, dim=1)
        return weights

class ProposedDualQNodeGating(nn.Module):
    def __init__(self, in_features=512, num_classes=4, n_qubits=6, n_layers=1):
        super().__init__()
        self.n_qubits = n_qubits
        self.pre_encoder = PreEncoder(in_features, n_qubits)
        self.qnn1, self.qnn2 = build_dual_qnodes(n_qubits=n_qubits, n_layers=n_layers)
        self.gate = GatingNetwork(in_features=n_qubits, hidden_dim=32, num_experts=2)
        self.bn_fusion = nn.BatchNorm1d(n_qubits)
        self.head = ClassifierHead(in_features=n_qubits, num_classes=num_classes)

    def forward(self, x, return_gates=False):
        x_cls = self.pre_encoder(x)
        f1 = self.qnn1(x_cls)
        f2 = self.qnn2(x_cls)
        w = self.gate(x_cls)
        fused = w[:, :1] * f1 + w[:, 1:] * f2
        z = self.bn_fusion(fused)
        logits = self.head(z)
        if return_gates:
            return logits, w
        return logits

class SymmetricalClassicalGating(nn.Module):
    def __init__(self, in_features=512, num_classes=4, n_qubits=6):
        super().__init__()
        self.n_qubits = n_qubits
        self.pre_encoder = PreEncoder(in_features, n_qubits)
        self.branch1 = nn.Sequential(
            nn.Linear(n_qubits, n_qubits, bias=False),
            nn.Tanh()
        )
        self.branch2 = nn.Sequential(
            nn.Linear(n_qubits, n_qubits, bias=False),
            nn.Tanh()
        )
        self.gate = GatingNetwork(in_features=n_qubits, hidden_dim=32, num_experts=2)
        self.bn_fusion = nn.BatchNorm1d(n_qubits)
        self.head = ClassifierHead(in_features=n_qubits, num_classes=num_classes)

    def forward(self, x, return_gates=False):
        x_cls = self.pre_encoder(x)
        f1 = self.branch1(x_cls)
        f2 = self.branch2(x_cls)
        w = self.gate(x_cls)
        fused = w[:, :1] * f1 + w[:, 1:] * f2
        z = self.bn_fusion(fused)
        logits = self.head(z)
        if return_gates:
            return logits, w
        return logits

class DualQNodeConcat(nn.Module):
    def __init__(self, in_features=512, num_classes=4, n_qubits=6, n_layers=1):
        super().__init__()
        self.pre_encoder = PreEncoder(in_features, n_qubits)
        self.qnn1, self.qnn2 = build_dual_qnodes(n_qubits=n_qubits, n_layers=n_layers)
        self.bn_fusion = nn.BatchNorm1d(2 * n_qubits)
        self.head = ClassifierHead(in_features=2 * n_qubits, num_classes=num_classes)

    def forward(self, x):
        x_cls = self.pre_encoder(x)
        f1 = self.qnn1(x_cls)
        f2 = self.qnn2(x_cls)
        fused = torch.cat([f1, f2], dim=1)
        z = self.bn_fusion(fused)
        return self.head(z)

class DualQNodeFixedSum(nn.Module):
    def __init__(self, in_features=512, num_classes=4, n_qubits=6, n_layers=1):
        super().__init__()
        self.pre_encoder = PreEncoder(in_features, n_qubits)
        self.qnn1, self.qnn2 = build_dual_qnodes(n_qubits=n_qubits, n_layers=n_layers)
        self.bn_fusion = nn.BatchNorm1d(n_qubits)
        self.head = ClassifierHead(in_features=n_qubits, num_classes=num_classes)

    def forward(self, x):
        x_cls = self.pre_encoder(x)
        f1 = self.qnn1(x_cls)
        f2 = self.qnn2(x_cls)
        fused = 0.5 * f1 + 0.5 * f2
        z = self.bn_fusion(fused)
        return self.head(z)

class SingleQNodeModel(nn.Module):
    def __init__(self, observable='Z', in_features=512, num_classes=4, n_qubits=6, n_layers=1):
        super().__init__()
        self.pre_encoder = PreEncoder(in_features, n_qubits)
        self.qnn = build_single_qnode(observable=observable, n_qubits=n_qubits, n_layers=n_layers)
        self.bn = nn.BatchNorm1d(n_qubits)
        self.head = ClassifierHead(in_features=n_qubits, num_classes=num_classes)

    def forward(self, x):
        x_cls = self.pre_encoder(x)
        f = self.qnn(x_cls)
        z = self.bn(f)
        return self.head(z)

class PureClassicalBottleneck(nn.Module):
    def __init__(self, in_features=512, num_classes=4, n_qubits=6):
        super().__init__()
        self.pre_encoder = PreEncoder(in_features, n_qubits)
        self.branch = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.Tanh()
        )
        self.bn = nn.BatchNorm1d(n_qubits)
        self.head = ClassifierHead(in_features=n_qubits, num_classes=num_classes)

    def forward(self, x):
        x_cls = self.pre_encoder(x)
        f = self.branch(x_cls)
        z = self.bn(f)
        return self.head(z)

class ClassicalConcatTwin(nn.Module):
    """Fully classical twin of DualQNodeConcat: two tanh Linear(6->6) branches,
    concatenated (12-d) into a plain classical head. Tests whether the quantum
    branches add anything over a classical capacity-matched concat pipeline."""
    def __init__(self, in_features=512, num_classes=4, n_qubits=6, n_layers=1):
        super().__init__()
        self.n_qubits = n_qubits
        self.pre_encoder = PreEncoder(in_features, n_qubits)
        self.branch1 = nn.Sequential(nn.Linear(n_qubits, n_qubits, bias=False), nn.Tanh())
        self.branch2 = nn.Sequential(nn.Linear(n_qubits, n_qubits, bias=False), nn.Tanh())
        self.bn_fusion = nn.BatchNorm1d(2 * n_qubits)
        self.head = ClassifierHead(in_features=2 * n_qubits, num_classes=num_classes)

    def forward(self, x):
        x_cls = self.pre_encoder(x)
        fused = torch.cat([self.branch1(x_cls), self.branch2(x_cls)], dim=1)
        z = self.bn_fusion(fused)
        return self.head(z)

class ProposedDualQNodeNoEnt(nn.Module):
    """Proposed Dual-QNode Gating with ALL entangling gates removed
    (no CZ in branch Z, no CNOT in branch X). Product-state circuits,
    identical 12 trainable angles."""
    def __init__(self, in_features=512, num_classes=4, n_qubits=6, n_layers=1):
        super().__init__()
        self.n_qubits = n_qubits
        self.pre_encoder = PreEncoder(in_features, n_qubits)
        self.qnn1, self.qnn2 = build_dual_qnodes(n_qubits=n_qubits, n_layers=n_layers,
                                                 entangle_z=False, entangle_x=False)
        self.gate = GatingNetwork(in_features=n_qubits, hidden_dim=32, num_experts=2)
        self.bn_fusion = nn.BatchNorm1d(n_qubits)
        self.head = ClassifierHead(in_features=n_qubits, num_classes=num_classes)

    def forward(self, x, return_gates=False):
        x_cls = self.pre_encoder(x)
        f1 = self.qnn1(x_cls)
        f2 = self.qnn2(x_cls)
        w = self.gate(x_cls)
        fused = w[:, :1] * f1 + w[:, 1:] * f2
        z = self.bn_fusion(fused)
        logits = self.head(z)
        if return_gates:
            return logits, w
        return logits

class ProposedDualQNodeNoCNOT(nn.Module):
    """Proposed Dual-QNode Gating with only the CNOT entanglement removed
    (branch Z keeps CZ). Isolates the entanglement topology that makes the
    transverse branch input-independent (see experiments/debug_qnode_x_collapse.py)."""
    def __init__(self, in_features=512, num_classes=4, n_qubits=6, n_layers=1):
        super().__init__()
        self.n_qubits = n_qubits
        self.pre_encoder = PreEncoder(in_features, n_qubits)
        self.qnn1, self.qnn2 = build_dual_qnodes(n_qubits=n_qubits, n_layers=n_layers,
                                                 entangle_z=True, entangle_x=False)
        self.gate = GatingNetwork(in_features=n_qubits, hidden_dim=32, num_experts=2)
        self.bn_fusion = nn.BatchNorm1d(n_qubits)
        self.head = ClassifierHead(in_features=n_qubits, num_classes=num_classes)

    def forward(self, x, return_gates=False):
        x_cls = self.pre_encoder(x)
        f1 = self.qnn1(x_cls)
        f2 = self.qnn2(x_cls)
        w = self.gate(x_cls)
        fused = w[:, :1] * f1 + w[:, 1:] * f2
        z = self.bn_fusion(fused)
        logits = self.head(z)
        if return_gates:
            return logits, w
        return logits

class ProposedDualQNodeNoisy(nn.Module):
    """Proposed Dual-QNode Gating under a parameterized NISQ noise model:
    depolarizing after state prep, depolarizing after every gate, and
    bit-flip readout error before measurement. Quantum branches are executed
    by the batched torch density-matrix simulator (validated against
    PennyLane default.mixed in experiments/validate_torch_sim.py)."""
    def __init__(self, in_features=512, num_classes=4, n_qubits=6, n_layers=1,
                 p_prep=0.0, p_gate=0.0, p_meas=0.0):
        super().__init__()
        self.n_qubits = n_qubits
        self.noise_params = {'p_prep': p_prep, 'p_gate': p_gate, 'p_meas': p_meas}
        self.pre_encoder = PreEncoder(in_features, n_qubits)
        self.qnn = TorchNoisyDualQNode(n_qubits=n_qubits, p_prep=p_prep,
                                       p_gate=p_gate, p_meas=p_meas,
                                       dtype=torch.complex64)
        self.gate = GatingNetwork(in_features=n_qubits, hidden_dim=32, num_experts=2)
        self.bn_fusion = nn.BatchNorm1d(n_qubits)
        self.head = ClassifierHead(in_features=n_qubits, num_classes=num_classes)

    def forward(self, x, return_gates=False):
        x_cls = self.pre_encoder(x)
        f1, f2 = self.qnn(x_cls)
        f1, f2 = f1.float(), f2.float()
        w = self.gate(x_cls)
        fused = w[:, :1] * f1 + w[:, 1:] * f2
        z = self.bn_fusion(fused)
        logits = self.head(z)
        if return_gates:
            return logits, w
        return logits

class ProposedDualQNodeNoisyPL(ProposedDualQNodeNoisy):
    """PennyLane default.mixed backend variant — kept only as the reference
    implementation for simulator validation (slow, CPU)."""
    def __init__(self, in_features=512, num_classes=4, n_qubits=6, n_layers=1,
                 p_prep=0.0, p_gate=0.0, p_meas=0.0, diff_method='backprop'):
        nn.Module.__init__(self)
        self.n_qubits = n_qubits
        self.noise_params = {'p_prep': p_prep, 'p_gate': p_gate, 'p_meas': p_meas}
        self.pre_encoder = PreEncoder(in_features, n_qubits)
        self.qnn1, self.qnn2 = build_dual_qnodes_noisy(n_qubits=n_qubits, n_layers=n_layers,
                                                       p_prep=p_prep, p_gate=p_gate,
                                                       p_meas=p_meas, diff_method=diff_method)
        self.gate = GatingNetwork(in_features=n_qubits, hidden_dim=32, num_experts=2)
        self.bn_fusion = nn.BatchNorm1d(n_qubits)
        self.head = ClassifierHead(in_features=n_qubits, num_classes=num_classes)

    def forward(self, x, return_gates=False):
        x_cls = self.pre_encoder(x)
        f1 = self.qnn1(x_cls)
        f2 = self.qnn2(x_cls)
        w = self.gate(x_cls)
        fused = w[:, :1] * f1 + w[:, 1:] * f2
        z = self.bn_fusion(fused)
        logits = self.head(z)
        if return_gates:
            return logits, w
        return logits

class CosineBottleneckControl(nn.Module):
    """6-parameter classical cosine control baseline requested by Reviewer 3.
    Directly mirrors the closed-form expectation of QNode-Z:
        f_k(x) = a_k * cos(x_k), where a in R^{1 x 6} are 6 trainable parameters.
    Matches SingleQNodeZ (5,960 params) exactly."""
    def __init__(self, in_features=512, num_classes=4, n_qubits=6, n_layers=1, **kwargs):
        super().__init__()
        self.pre_encoder = PreEncoder(in_features, n_qubits)
        self.a = nn.Parameter(torch.ones(1, n_qubits)) # Initialized to 1.0 (equivalent to theta=0)
        self.bn = nn.BatchNorm1d(n_qubits)
        self.head = ClassifierHead(in_features=n_qubits, num_classes=num_classes)

    def forward(self, x):
        x_cls = self.pre_encoder(x)
        f = self.a * torch.cos(x_cls)
        z = self.bn(f)
        return self.head(z)

MODEL_REGISTRY = {
    'proposed_dual_qnode': ProposedDualQNodeGating,
    'symmetrical_classical': SymmetricalClassicalGating,
    'dual_qnode_concat': DualQNodeConcat,
    'dual_qnode_fixed_sum': DualQNodeFixedSum,
    'single_qnode_z': lambda in_f, num_c, n_q, n_l: SingleQNodeModel('Z', in_f, num_c, n_q, n_l),
    'single_qnode_x': lambda in_f, num_c, n_q, n_l: SingleQNodeModel('X', in_f, num_c, n_q, n_l),
    'pure_classical': PureClassicalBottleneck,
    'cosine_bottleneck': CosineBottleneckControl,
    # Tier-2 additions
    'proposed_noent': ProposedDualQNodeNoEnt,
    'proposed_no_cnot': ProposedDualQNodeNoCNOT,
    'classical_concat': ClassicalConcatTwin,
    'proposed_noisy_short': lambda in_features, num_classes, n_qubits, n_layers: ProposedDualQNodeNoisy(
        in_features, num_classes, n_qubits, n_layers, p_prep=1e-3, p_gate=1e-3, p_meas=1e-2),
    'proposed_noisy_expected': lambda in_features, num_classes, n_qubits, n_layers: ProposedDualQNodeNoisy(
        in_features, num_classes, n_qubits, n_layers, p_prep=1e-4, p_gate=1e-4, p_meas=1e-3),
    # reference (slow) backend, validation only
    'proposed_noisy_pl': lambda in_features, num_classes, n_qubits, n_layers: ProposedDualQNodeNoisyPL(
        in_features, num_classes, n_qubits, n_layers, p_prep=1e-3, p_gate=1e-3, p_meas=1e-2),
}

def build_model(model_name, in_features=512, num_classes=4, n_qubits=6, n_layers=1):
    if model_name not in MODEL_REGISTRY:
        raise ValueError('Unknown model: ' + model_name)
    cls = MODEL_REGISTRY[model_name]
    if model_name in ['single_qnode_z', 'single_qnode_x']:
        model = cls(in_features, num_classes, n_qubits, n_layers)
    elif model_name in ['symmetrical_classical', 'pure_classical']:
        model = cls(in_features=in_features, num_classes=num_classes, n_qubits=n_qubits)
    else:
        model = cls(in_features=in_features, num_classes=num_classes, n_qubits=n_qubits, n_layers=n_layers)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, trainable_params

if __name__ == '__main__':
    dummy_x = torch.randn(4, 512)
    for name in MODEL_REGISTRY.keys():
        m, p_cnt = build_model(name, in_features=512, num_classes=4)
        out = m(dummy_x)
        print(f'[OK] {name:<25} | Params: {p_cnt:>5} | Out Shape: {list(out.shape)}')
