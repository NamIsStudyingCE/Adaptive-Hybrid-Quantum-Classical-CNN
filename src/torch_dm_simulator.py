"""Batched torch density-matrix simulator for the noisy Dual-QNode branches.

Replicates EXACTLY the operation order of src/circuits.build_dual_qnodes_noisy
(PennyLane default.mixed): R_X angle embedding -> per-wire depolarizing (prep)
-> ansatz (RY/RZ rotations + CZ/CNOT ring, depolarizing after every gate) ->
per-wire bit-flip (readout) -> Pauli expectations.

Speed rationale: the whole batch of density matrices [B, 64, 64] is advanced
with dense complex matmuls on one device (CUDA), with no per-gate Python
dispatch. Validated against PennyLane default.mixed in
experiments/validate_torch_sim.py before any benchmark run.
"""
import math
import torch
import torch.nn as nn


class TorchNoisyDualQNode(nn.Module):
    """Returns (f1, f2): sigma_z and sigma_x expectation branches, each [B, 6].

    Trainable parameters: weights1, weights2, each shape (1, n_qubits, 1) —
    identical layout to PennyLane TorchLayer so parameter counts match (12).
    """

    def __init__(self, n_qubits=6, p_prep=0.0, p_gate=0.0, p_meas=0.0,
                 dtype=torch.complex128):
        super().__init__()
        self.n_qubits = n_qubits
        self.p_prep, self.p_gate, self.p_meas = float(p_prep), float(p_gate), float(p_meas)
        self.dtype = dtype
        self.real_dtype = {torch.complex128: torch.float64,
                           torch.complex64: torch.float32}[dtype]
        dim = 2 ** n_qubits

        def kron_op(op2, wire):
            mats = [torch.eye(2, dtype=torch.complex128) for _ in range(n_qubits)]
            mats[wire] = op2.to(torch.complex128)
            out = mats[0]
            for m in mats[1:]:
                out = torch.kron(out, m)
            return out

        PX = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex128)
        PY = torch.tensor([[0., -1j], [1j, 0.]], dtype=torch.complex128)
        PZ = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex128)

        for i in range(n_qubits):
            self.register_buffer(f'X{i}', kron_op(PX, i).to(dtype))
            self.register_buffer(f'Y{i}', kron_op(PY, i).to(dtype))
            self.register_buffer(f'Z{i}', kron_op(PZ, i).to(dtype))

        # wire 0 = leftmost kron factor (PennyLane big-endian convention)
        even_pairs = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]
        odd_pairs = [(i, i + 1) for i in range(1, n_qubits - 1, 2)]
        ring_close = [(n_qubits - 1, 0)] if n_qubits > 1 else []
        # gate order in ansatz_z/ansatz_x: even CZ/CNOT pairs -> odd rotations
        # happen between -> odd pairs -> ring close. Store gates in one list and
        # index them in exactly that order:
        all_pairs = even_pairs + odd_pairs + ring_close
        for k, (a, b) in enumerate(all_pairs):
            self.register_buffer(f'CZ{k}', self._cz(a, b, dim).to(dtype))
            self.register_buffer(f'CN{k}', self._cnot(a, b, dim).to(dtype))
        self.n_even_pairs = len(even_pairs)
        self.n_odd_pairs = len(odd_pairs)
        self.even_wires = list(range(0, n_qubits, 2))
        self.odd_wires = list(range(1, n_qubits, 2))

        self.weights1 = nn.Parameter(torch.empty(1, n_qubits, 1))
        self.weights2 = nn.Parameter(torch.empty(1, n_qubits, 1))
        nn.init.uniform_(self.weights1, 0.0, 2 * math.pi)
        nn.init.uniform_(self.weights2, 0.0, 2 * math.pi)

    def _cz(self, a, b, dim):
        d = torch.ones(dim, dtype=torch.complex128)
        for s in range(dim):
            ba = (s >> (self.n_qubits - 1 - a)) & 1
            bb = (s >> (self.n_qubits - 1 - b)) & 1
            if ba == 1 and bb == 1:
                d[s] = -1.0
        return torch.diag(d)

    def _cnot(self, control, target, dim):
        idx = torch.arange(dim)
        map_idx = idx.clone()
        for s in range(dim):
            bc = (s >> (self.n_qubits - 1 - control)) & 1
            if bc == 1:
                map_idx[s] = s ^ (1 << (self.n_qubits - 1 - target))
        U = torch.zeros(dim, dim, dtype=torch.complex128)
        U[map_idx, idx] = 1.0
        return U

    # ---- core ops ----
    def _rot_scalar(self, rho, P, theta):
        """U = cos(t/2) I - i sin(t/2) P; theta = scalar tensor."""
        theta = theta.to(self.real_dtype)
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        Prho = torch.matmul(P, rho)
        rhoP = torch.matmul(rho, P)
        return (c * c) * rho + (s * s) * torch.matmul(Prho, P) \
            + (1j * s * c) * (rhoP - Prho)

    def _rot_batch(self, rho, P, theta):
        """U(x_i) per sample; theta = tensor [B]."""
        theta = theta.to(self.real_dtype)
        c = torch.cos(theta / 2).view(-1, 1, 1)
        s = torch.sin(theta / 2).view(-1, 1, 1)
        Prho = torch.matmul(P, rho)
        rhoP = torch.matmul(rho, P)
        return (c * c) * rho + (s * s) * torch.matmul(Prho, P) \
            + (1j * s * c) * (rhoP - Prho)

    def _dep_wire(self, rho, i):
        """Depolarizing on wire i: (1-p)rho + p/3 (XrhoX + YrhoY + ZrhoZ)."""
        p = self.p_gate
        out = (1 - p) * rho
        for tag in ('X', 'Y', 'Z'):
            P = getattr(self, f'{tag}{i}')
            out = out + (p / 3) * torch.matmul(P, torch.matmul(rho, P))
        return out

    def _expect(self, rho, i, tag):
        O = getattr(self, f'{tag}{i}')
        if tag == 'Z':
            zdiag = torch.diagonal(O)
            return torch.real(torch.diagonal(rho, dim1=-2, dim2=-1) @ zdiag)
        return torch.real((rho * O.T).sum(dim=(-2, -1)))

    def _entangle(self, rho, gate_idx, a, b, tag):
        G = getattr(self, ('CZ' if tag == 'z' else 'CN') + str(gate_idx))
        rho = torch.matmul(G, torch.matmul(rho, G.conj().T))
        if self.p_gate > 0:
            rho = self._dep_wire(rho, a)
            rho = self._dep_wire(rho, b)
        return rho

    def _branch(self, x, weights, tag):
        B = x.shape[0]
        device = x.device
        # initial state |0...0><0...0| (pure zero state, NOT the identity!)
        rho0 = torch.zeros(2 ** self.n_qubits, 2 ** self.n_qubits,
                           dtype=self.dtype, device=device)
        rho0[0, 0] = 1.0
        rho = rho0.unsqueeze(0).expand(B, -1, -1).contiguous()
        # R_X angle embedding, per-sample angles
        for i in range(self.n_qubits):
            rho = self._rot_batch(rho, getattr(self, f'X{i}'), x[:, i].to(self.real_dtype))
        # prep depolarizing per wire
        if self.p_prep > 0:
            for i in range(self.n_qubits):
                rho = self._dep_wire(rho, i)

        def rotate(wire, theta):
            nonlocal rho
            Pw = getattr(self, ('Y' if tag == 'z' else 'Z') + str(wire))
            rho = self._rot_scalar(rho, Pw, theta)
            if self.p_gate > 0:
                rho = self._dep_wire(rho, wire)

        # even-wire rotations
        for i in self.even_wires:
            rotate(i, weights[i])
        # even entangling pairs = gates [0 .. n_even_pairs)
        for k in range(self.n_even_pairs):
            a, b = (list(range(0, self.n_qubits - 1, 2))[k], list(range(0, self.n_qubits - 1, 2))[k] + 1)
            rho = self._entangle(rho, k, a, b, tag)
        # odd-wire rotations
        for i in self.odd_wires:
            rotate(i, weights[i])
        # odd entangling pairs
        odd_pair_list = [(i, i + 1) for i in range(1, self.n_qubits - 1, 2)]
        for k, (a, b) in enumerate(odd_pair_list):
            rho = self._entangle(rho, self.n_even_pairs + k, a, b, tag)
        # ring close
        if self.n_qubits > 1:
            rho = self._entangle(rho, self.n_even_pairs + self.n_odd_pairs,
                                 self.n_qubits - 1, 0, tag)
        # readout bit-flip
        if self.p_meas > 0:
            for i in range(self.n_qubits):
                p = self.p_meas
                X = getattr(self, f'X{i}')
                rho = (1 - p) * rho + p * torch.matmul(X, torch.matmul(rho, X))
        return torch.stack(
            [self._expect(rho, i, 'Z' if tag == 'z' else 'X') for i in range(self.n_qubits)],
            dim=1)

    def forward(self, x):
        f1 = self._branch(x, self.weights1[0, :, 0], 'z')
        f2 = self._branch(x, self.weights2[0, :, 0], 'x')
        return f1, f2
