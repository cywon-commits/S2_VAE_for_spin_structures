"""Task C: LL model with quadrupolar field  H = -sum P2(n_i.n_j) - eps sum P2(n_i.d), d=z.
Bingham heat bath: add (3/2) beta eps d d^T to A.  S(eps)=<P2(n.d)>."""
import sys, json, time, numpy as np
sys.path.insert(0, "/home/user/S2_VAE_for_spin_structures/src")
from ll_mc import batched_bingham
def run(L, T, eps, n_eq, n_meas, seed, every=5):
    rng = np.random.default_rng(seed)
    n = np.tile([0, 0, 1.0], (L, L, L, 1)) + 0.05 * rng.normal(size=(L, L, L, 3)); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    X, Y, Z = np.meshgrid(*[np.arange(L)] * 3, indexing='ij'); masks = [((X + Y + Z) % 2) == p for p in (0, 1)]
    beta = 1 / T; D = np.zeros((3, 3)); D[2, 2] = 1.0; S = []
    for sw in range(n_eq + n_meas):
        for mk in masks:
            Ms = np.zeros((L, L, L, 3, 3))
            for ax in range(3):
                for s in (1, -1):
                    m = np.roll(n, s, ax); Ms += m[..., :, None] * m[..., None, :]
            A = 1.5 * beta * (Ms[mk] + eps * D)
            n[mk] = batched_bingham(A, rng)
        if sw >= n_eq and (sw - n_eq) % every == 0:
            S.append(np.mean(1.5 * n[..., 2] ** 2 - 0.5))
    return np.array(S)
if __name__ == "__main__":
    L = 12; out = {}; t0 = time.time()
    for T in (0.95, 1.05, 1.10):
        for eps in (0, 0.01, 0.02, 0.05, 0.1, 0.2):
            ss = []
            for seed in range(4):
                s = run(L, T, eps, 400, 2000, 7000 + seed + int(1000 * T)); ss.append(s.mean())
            out[f"{T}_{eps}"] = dict(T=T, eps=eps, S=float(np.mean(ss)), se=float(np.std(ss, ddof=1) / 2), per_seed=[float(x) for x in ss])
            print(T, eps, out[f"{T}_{eps}"]["S"], out[f"{T}_{eps}"]["se"], f"{time.time()-t0:.0f}s", flush=True)
            json.dump(out, open("/home/user/S2_VAE_for_spin_structures/results/response/C_quad.json", "w"), indent=1)
