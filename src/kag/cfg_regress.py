"""E3 v2 C: y = N [NLL(R=1) - NLL(R)] per validation config regressed on N*g_ring (ring oracle, per config) and n_def (3x3 block phase defects):  y = a*(N g_ring) + b*n_def + c.
Standard errors clustered by validation seed (100 snapshots per seed).  python3 cfg_regress.py out.json kind R [T ...]    kind = v1 (kagu_*) | k6 (kag6_*)"""
import sys, json, os, numpy as np
from eval_ku import per_config
from eval_k6 import per_config6
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "../../results/kag/"
def arrays(kind, T, R):
    f = f"{D}cfgarr_{kind}_T{T:.4f}_R{R}.npz"
    if os.path.exists(f): z = np.load(f); return z["nll"], z["nll_base"]
    base = per_config(f"{D}kag_T{T:.4f}_R1_p.pt")["nll_cfg"].numpy()
    r = (per_config(f"{D}kagu_T{T:.4f}_R{R}.pt") if kind == "v1" else per_config6(f"{D}kag6_T{T:.4f}_R{R}.pt"))["nll_cfg"].numpy(); np.savez(f, nll=r, nll_base=base); return r, base
def ols(X, y, cl):
    n, p = X.shape; XtXi = np.linalg.inv(X.T @ X); beta = XtXi @ X.T @ y; e = y - X @ beta; meat = np.zeros((p, p))
    for c in np.unique(cl): s = X[cl == c].T @ e[cl == c]; meat += np.outer(s, s)
    G = len(np.unique(cl)); V = XtXi @ meat @ XtXi * G / (G - 1); r2 = 1 - e.var() / y.var(); return beta, np.sqrt(np.diag(V)), r2
if __name__ == "__main__":
    kind, R = sys.argv[2], int(sys.argv[3]); Ts = [float(t) for t in sys.argv[4:]]; N = 432; rows = []; data = {}
    for T in Ts:
        nll, base = arrays(kind, T, R); y = base - nll; z = json.load(open(f"{D}cfgterms_T{T:.4f}.json")); g = N * np.array(z["g_ring"]); nd = np.array(z["n_def"], float); cl = np.repeat(np.arange(8), len(y) // 8); data[T] = (g, nd, y)
        use_def = nd.std() > 0 and T > 0.0006
        X = np.stack([g] + ([nd] if use_def else []) + [np.ones_like(g)], 1); b, s, r2 = ols(X, y, cl); names = ["a"] + (["b"] if use_def else []) + ["c"]
        rows.append(dict(T=T, R=R, kind=kind, y_mean=float(y.mean()), r2=float(r2), n_def_mean=float(nd.mean()), **{k: float(v) for k, v in zip(names, b)}, **{k + "_se": float(v) for k, v in zip(names, s)}))
        print(rows[-1], flush=True)
    if len(Ts) > 1:
        g = np.concatenate([data[T][0] for T in Ts]); nd = np.concatenate([data[T][1] for T in Ts]); y = np.concatenate([data[T][2] for T in Ts]); cl = np.concatenate([np.repeat(np.arange(8) + 8 * i, 100) for i in range(len(Ts))])
        b, s, r2 = ols(np.stack([g, nd, np.ones_like(g)], 1), y, cl); rows.append(dict(T="pooled", R=R, kind=kind, a=float(b[0]), a_se=float(s[0]), b=float(b[1]), b_se=float(s[1]), c=float(b[2]), c_se=float(s[2]), r2=float(r2))); print(rows[-1])
    json.dump(rows, open(sys.argv[1], "w"), indent=1)
    fig, ax = plt.subplots(1, len(Ts), figsize=(4.5 * len(Ts), 4), squeeze=False)
    for a_, T in zip(ax[0], Ts):
        g, nd, y = data[T]; sc = a_.scatter(g, y, c=nd, s=8, cmap="viridis"); a_.plot([g.min(), g.max()], [g.min(), g.max()], "k--", lw=0.8); a_.set_xlabel("N g_ring (nats)"); a_.set_ylabel("N [NLL(R=1) - NLL(R)] (nats)"); a_.set_title(f"T={T}, R={R} ({kind})"); plt.colorbar(sc, ax=a_, label="n_def")
    plt.tight_layout(); plt.savefig(sys.argv[1].replace(".json", ".png"), dpi=110)
