"""Kagome Heisenberg AF in a uniform field: H = J sum n_i.n_j - h.sum n_i (J=1).
Heat bath with local field -sum_nb n + h, energy-conserving over-relaxation about the same local field.
Usage: python kagome_mc_field.py L T hz n_anneal n_meas n_or seed  -> prints m_z, chi from fluctuations, <(nu.z)^2>"""
import numpy as np, sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kagome"))
from kagome_mc import nbsum, heatbath

def run(L, T, hvec, n_anneal, n_meas, n_or, seed, every=5):
    rng = np.random.default_rng(seed)
    n = rng.normal(size=(L, L, 3, 3)); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    hvec = np.asarray(hvec, float)
    Ts = np.geomspace(0.1, T, n_anneal)
    obs = []
    for sw in range(n_anneal + n_meas):
        Tc = Ts[sw] if sw < n_anneal else T
        for s in range(3):
            f = -nbsum(n, s) + hvec
            n[:, :, s] = heatbath(f / Tc, rng)
        for _ in range(n_or):
            for s in range(3):
                f = -nbsum(n, s) + hvec
                fh = f / np.linalg.norm(f, axis=-1, keepdims=True)
                n[:, :, s] = 2 * np.sum(n[:, :, s] * fh, -1, keepdims=True) * fh - n[:, :, s]
        if sw >= n_anneal and (sw - n_anneal) % every == 0:
            S = n.reshape(-1, 3); M = S.sum(0)
            w, v = np.linalg.eigh(S.T @ S); nu = v[:, 0]
            E = sum(np.sum(n[:, :, s] * nbsum(n, s)) for s in range(3)) / 2 - hvec @ M
            obs.append(np.r_[M, (nu @ hvec / max(np.linalg.norm(hvec), 1e-12)) ** 2, w[0] / len(S), E / len(S)])
    return np.array(obs), L * L * 3

if __name__ == "__main__":
    L, T, hz = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    na, nm, nor, seed = map(int, sys.argv[4:8])
    O, N = run(L, T, [0, 0, hz], na, nm, nor, seed)
    M = O[:, :3]
    out = dict(L=L, T=T, h=hz, seed=seed, m=float(M[:, 2].mean() / N), mx=float(M[:, 0].mean() / N),
               chi_zz_fluct=float(M[:, 2].var() / N / T), chi_xx_fluct=float(((M[:, 0] ** 2 + M[:, 1] ** 2) / 2).mean() / N / T),
               nu_h2=float(O[:, 3].mean()), oop=float(O[:, 4].mean()), e=float(O[:, 5].mean()), nmeas=len(O))
    print(json.dumps(out))
