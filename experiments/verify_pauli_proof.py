# Verify teacher's Pauli-string proof:
#   Lemma: post-RX-embedding, every Pauli string containing >=1 X has <P> = 0 exactly.
#   Proposition (brick-layer): U^dag X_k U expands ONLY into X-containing strings (0 X-free terms, all 6 wires).
#   Sequential: exactly 1 X-free string per wire (the surviving information).
import itertools
import numpy as np

n = 6
dim = 2 ** n
I2 = np.eye(2)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_op(factors):
    out = np.array([[1]], dtype=complex)
    for f in factors:
        out = np.kron(out, f)
    return out

def rx(x):
    return np.array([[np.cos(x / 2), -1j * np.sin(x / 2)],
                     [-1j * np.sin(x / 2), np.cos(x / 2)]])

rng = np.random.default_rng(11)
x = rng.uniform(0, 3, n)

# ---- Lemma: expectation of all 4096 strings on the post-embedding product state
labels = ['I', 'X', 'Y', 'Z']
mats = {'I': I2, 'X': PX, 'Y': PY, 'Z': PZ}
emb_1q = [rx(x[i]) @ np.outer([1, 0], [1, 0]) @ rx(x[i]).conj().T for i in range(n)]

x_containing_nonzero = 0
total_x_strings = 0
for combo in itertools.product('IXYZ', repeat=n):
    has_x = 'X' in combo
    P = kron_op([mats[c] for c in combo])
    # expectation = sum over factorized terms; product-state expectation:
    expval = 1.0
    for i, c in enumerate(combo):
        expval *= np.trace(emb_1q[i] @ mats[c]).real
    if has_x:
        total_x_strings += 1
        if abs(expval) > 1e-12:
            x_containing_nonzero += 1
print(f"LEMMA: X-containing strings = {total_x_strings}/4096; nonzero expectation = {x_containing_nonzero} (expect 0)")

# ---- Build ansatz unitaries (no embedding) as 64x64 matrices
def rz(phi):
    return np.array([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]], dtype=complex)

def cnot(c, t):
    m = np.zeros((dim, dim), dtype=complex)
    for s in range(dim):
        b = [(s >> (n - 1 - w)) & 1 for w in range(n)]
        if b[c] == 1:
            b[t] ^= 1
        t_idx = sum(bit << (n - 1 - w) for w, bit in enumerate(b))
        m[t_idx, s] = 1
    return m

phi = rng.uniform(0, 2 * np.pi, n)
CNOT_pairs_even = [(0, 1), (2, 3), (4, 5)]
CNOT_pairs_odd = [(1, 2), (3, 4), (5, 0)]

def build_u(schedule):
    U = np.eye(dim, dtype=complex)
    for kind, wires_ in schedule:
        if kind == 'rz':
            U = U @ kron_op([rz(phi[w]) if w in wires_ else I2 for w in range(n)])
        else:
            for (c, t) in wires_:
                U = U @ cnot(c, t)
    return U

U_brick = build_u([('rz', [0, 2, 4]), ('cnot', CNOT_pairs_even), ('rz', [1, 3, 5]), ('cnot', CNOT_pairs_odd + [(5, 0)])])
U_seq = build_u([('rz', list(range(n))), ('cnot', CNOT_pairs_even + CNOT_pairs_odd + [(5, 0)])])

# ---- Pauli expansion of O_k = U^dag X_k U: count X-free strings with |coeff| > tol
pauli_list = []
for combo in itertools.product('IXYZ', repeat=n):
    P = kron_op([mats[c] for c in combo])
    pauli_list.append((combo, P))

def count_xfree(U, k, tol=1e-10):
    O = U.conj().T @ kron_op([PX if i == k else I2 for i in range(n)]) @ U
    xfree = 0
    max_free_coeff = 0.0
    for combo, P in pauli_list:
        if 'X' in combo:
            continue  # contains X
        coeff = np.trace(P @ O) / dim
        if abs(coeff) > tol:
            xfree += 1
            max_free_coeff = max(max_free_coeff, abs(coeff))
    return xfree, max_free_coeff

print()
print("PROPOSITION: X-free strings in U^dag X_k U (tol 1e-10)")
for k in range(n):
    b, mb = count_xfree(U_brick, k)
    s_, ms = count_xfree(U_seq, k)
    print(f"  wire {k}: brick-layer X-free = {b} (expect 0) | sequential X-free = {s_} (expect 1), max coeff = {ms:.4f}")
