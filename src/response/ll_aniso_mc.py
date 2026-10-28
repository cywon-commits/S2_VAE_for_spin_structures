"""E6 task C: LL MC with single-ion anisotropy  dH = -D sum P2(n_i.e), e = z.  Heat bath: A += 1.5*beta*D*e e^T.  Random start.
Observables per snapshot (director d = top eigvec of Q): S=<P2(n.d)>, A=P2(d.e), B=<(n.e)^2-(n.e')^2> with e'=d x e/|d x e|.
python3 ll_aniso_mc.py T seed_list -> results/response/C_aniso_T{T}.jsonl"""
import sys, json, time, numpy as np
sys.path.insert(0, "/home/user/S2_VAE_for_spin_structures/src")
from ll_mc import batched_bingham
DS = [0, 0.002, -0.002, 0.005, -0.005, 0.01, -0.01, 0.02, -0.02, 0.05, -0.05]
def run(L, T, D, n_eq, n_meas, seed, every=5):
    rng = np.random.default_rng(seed); n = rng.normal(size=(L, L, L, 3)); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    X, Y, Z = np.meshgrid(*[np.arange(L)] * 3, indexing='ij'); masks = [((X + Y + Z) % 2) == p for p in (0, 1)]
    beta = 1 / T; E = np.zeros((3, 3)); E[2, 2] = 1.0; obs = []; ez = np.array([0, 0, 1.0])
    for sw in range(n_eq + n_meas):
        for mk in masks:
            Ms = np.zeros((L, L, L, 3, 3))
            for ax in range(3):
                for s in (1, -1):
                    m = np.roll(n, s, ax); Ms += m[..., :, None] * m[..., None, :]
            n[mk] = batched_bingham(1.5 * beta * (Ms[mk] + D * E), rng)
        if sw >= n_eq and (sw - n_eq) % every == 0:
            Xf = n.reshape(-1, 3); Q = 1.5 * (Xf.T @ Xf) / len(Xf) - 0.5 * np.eye(3); w, V = np.linalg.eigh(Q); d = V[:, 2]
            c = np.cross(d, ez); nc = np.linalg.norm(c); ep = c / nc if nc > 1e-6 else np.array([1.0, 0, 0])
            obs.append([np.mean(1.5 * (Xf @ d) ** 2 - 0.5), 1.5 * d[2] ** 2 - 0.5, np.mean((Xf[:, 2]) ** 2 - (Xf @ ep) ** 2)])
    return np.array(obs)
if __name__ == "__main__":
    T = float(sys.argv[1]); seeds = [int(x) for x in sys.argv[2].split(",")]; t0 = time.time()
    for seed in seeds:
        for D in DS:
            o = run(12, T, D, 1000, 3000, 100 * seed + int(1000 * T) + int(10000 * abs(D)) + (7 if D < 0 else 0))
            r = dict(T=T, D=D, seed=seed, S=float(o[:, 0].mean()), A=float(o[:, 1].mean()), B=float(o[:, 2].mean()), S_all=o[:, 0].tolist() if False else None)
            open(f"/home/user/S2_VAE_for_spin_structures/results/response/C_aniso_T{T}.jsonl", "a").write(json.dumps(r) + "\n"); print(T, seed, D, round(r["S"], 3), round(r["A"], 3), round(r["B"], 3), f"{time.time()-t0:.0f}s", flush=True)
