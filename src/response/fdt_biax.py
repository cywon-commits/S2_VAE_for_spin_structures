"""MC reference for E6: biaxial (m=+-2, director frame) susceptibility of the LL nematic.
Frame: d = top eigenvector of Q per snapshot; e1 = lab x projected on the plane normal to d (y if degenerate), e2 = d x e1 (independent of the in-plane
eigenvectors, so <B> = <C> = 0 by symmetry).  B = (n.e1)^2-(n.e2)^2, C = 2(n.e1)(n.e2).
chi_B = (beta/N) <B_tot^2 + C_tot^2>/2 ; chi_site = beta <B_i^2 + C_i^2>/2 (independent-site value); also chi_Q (m=0) for comparison."""
import numpy as np, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = {}
for T in (0.80, 0.95, 1.05, 1.10):
    S = np.load(f"{ROOT}/data/ll_L12_T{T:.3f}_o.npy").reshape(-1, 12 ** 3, 3).astype(float); N = S.shape[1]; b = 1 / T
    rows = []
    for X in S:
        Q = 1.5 * X.T @ X / N - 0.5 * np.eye(3); w, V = np.linalg.eigh(Q); d = V[:, 2]
        t = np.array([1., 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1., 0]); e1 = t - (t @ d) * d; e1 /= np.linalg.norm(e1); e2 = np.cross(d, e1)
        a1, a2, a0 = X @ e1, X @ e2, X @ d
        Bi, Ci, Pi = a1 ** 2 - a2 ** 2, 2 * a1 * a2, 1.5 * a0 ** 2 - 0.5
        rows.append([Bi.sum(), Ci.sum(), ((Bi ** 2 + Ci ** 2) / 2).mean(), Pi.sum(), Pi.var(), w[2]])
    r = np.array(rows); nb = 10; blk = np.array_split(np.arange(len(r)), nb)
    chiB = b / N * (r[:, 0] ** 2 + r[:, 1] ** 2) / 2
    chiB_b = [chiB[k].mean() for k in blk]
    chiQ = b / N * (r[:, 3] - r[:, 3].mean()) ** 2
    out[str(T)] = dict(T=T, n_snap=len(r), S=float(r[:, 5].mean()), chi_B=float(chiB.mean()), chi_B_se=float(np.std(chiB_b, ddof=1) / np.sqrt(nb)),
                       chi_B_site=float(b * r[:, 2].mean()), B_mean=float(r[:, 0].mean() / N), chi_Q=float(chiQ.mean()), chi_Q_site=float(b * r[:, 4].mean()))
    print(json.dumps(out[str(T)]))
json.dump(out, open(f"{ROOT}/src/response/fdt_biax.json", "w"), indent=1)
