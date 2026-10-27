"""H=0 kagome seeds: (i) fluctuation-dissipation chi = beta N <M_a M_b> in the body frame (e1,e2 in spin plane, nu normal);
(ii) frozen-Weiss-field (product decoder, latent fixed) chi from exact Langevin covariance of each site in its local field."""
import sys, glob, json, numpy as np
sys.path.insert(0, "/home/claude/paper/repo/src/kagome"); from kagome_mc import nbsum
def frame(X):
    Q = X.T @ X; w, V = np.linalg.eigh(Q); return V[:, [2, 1, 0]]          # columns e1, e2, nu
out = {}
for T in (0.0005, 0.001, 0.002, 0.004, 0.008):
    fs = sorted(glob.glob(f"/home/claude/paper/repo/data_kag/seeds/kag_L12_T{T:.4f}_s*.npy"))
    if not fs: continue
    per = []
    for f in fs:
        S = np.load(f).astype(float); N = S[0].size // 3; Cf, Cw = [], []
        for n in S:
            X = n.reshape(-1, 3); F = frame(X); M = X.sum(0) @ F; Cf.append(np.outer(M, M) / N)
            h = -np.stack([nbsum(n, s) for s in range(3)], 2).reshape(-1, 3); hm = np.linalg.norm(h, axis=1); a = h / hm[:, None] @ F; k = hm / T
            Lk = 1 / np.tanh(k) - 1 / k; Lp = 1 / k**2 - 1 / np.sinh(np.minimum(k, 700))**2
            C = (Lk / k)[:, None, None] * (np.eye(3) - a[:, :, None] * a[:, None, :]) + Lp[:, None, None] * a[:, :, None] * a[:, None, :]
            Cw.append(C.mean(0))
        per.append((np.mean(Cf, 0) / T, np.mean(Cw, 0) / T))
    F = np.array([p[0] for p in per]); W = np.array([p[1] for p in per])
    d = lambda A: (np.diagonal(A, axis1=1, axis2=2))
    r = dict(n_seeds=len(fs),
             fdt_inplane=float(d(F)[:, :2].mean()), fdt_inplane_se=float(d(F)[:, :2].mean(1).std(ddof=1) / np.sqrt(len(fs))),
             fdt_normal=float(d(F)[:, 2].mean()), fdt_normal_se=float(d(F)[:, 2].std(ddof=1) / np.sqrt(len(fs))),
             weiss_inplane=float(d(W)[:, :2].mean()), weiss_normal=float(d(W)[:, 2].mean()))
    out[T] = r; print(T, {k: round(v, 4) for k, v in r.items()}, flush=True)
json.dump(out, open("fdt_kag.json", "w"), indent=1)
