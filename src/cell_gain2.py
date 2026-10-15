"""Cell-frame oracle NLL gain (nats/site) at a given T, with the shuffled-spin overfit correction.  python3 cell_gain2.py T [nconf]
(same procedure as cell_gain.py; T and data path are arguments)"""
import sys, numpy as np
from fitlib import Fitter
T = float(sys.argv[1]); nconf = int(sys.argv[2]) if len(sys.argv) > 2 else 100
rng = np.random.default_rng(1)
def frames_global(S):
    Q = np.einsum('cxyzi,cxyzj->cij', S, S); return np.linalg.eigh(Q)[1][:, :, [1, 0, 2]]
def rotate_cells(S, b):
    c, L = S.shape[0], S.shape[1]; Sb = S.reshape(c, L // b, b, L // b, b, L // b, b, 3)
    Qb = np.einsum('cxaybzdi,cxaybzdj->cxyzij', Sb, Sb); V = np.linalg.eigh(Qb)[1][..., [1, 0, 2]]
    return np.einsum('cxaybzdi,cxyzij->cxaybzdj', Sb, V).reshape(-1, 3)
nll = lambda P: Fitter(4, order=59, alpha=1e-8).fit(P).nll
S = np.load(f'../data/ll_L12_T{T:.3f}_o.npy').astype(float)[:nconf]; N = 12 ** 3
base = nll(np.einsum('cxyzi,cij->cxyzj', S, frames_global(S)).reshape(-1, 3))
Ssh = S.reshape(len(S), -1, 3).copy()
for k in range(len(S)): Ssh[k] = Ssh[k][rng.permutation(N)]
Ssh = Ssh.reshape(S.shape); base_s = nll(np.einsum('cxyzi,cij->cxyzj', Ssh, frames_global(Ssh)).reshape(-1, 3))
print(f"T={T}: NLL gain from exact cell frames (nats/site), {nconf} configs")
for R, b in [(2, 6), (3, 4), (4, 3)]:
    g = base - nll(rotate_cells(S, b)); o = base_s - nll(rotate_cells(Ssh, b)); print(f" R={R} n={b**3:3d} raw {g:.4f} overfit {o:.4f} net {g-o:.4f}", flush=True)
