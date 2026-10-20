"""REQUEST_FIGURES_v2 B-3: why do the chat layered NLLs (A -1.214, B -1.695, C -1.949) differ from the recomputed ones (-0.933, -1.374, -1.579)?
Pipeline = src/layered/layered_gain.py (A, B) and layered_gain2.py (C with cell-frame loop axes), unchanged, but organised per seed.  python3 investigate_layered.py T   -> figures/data/layered_T{T}.json
Per seed (25 snapshots = every 4th): pooled spin arrays in the A / B / C local frames are stored (npz) so that fits can be repeated with other seed sets, L and quadrature."""
import sys, os, json, glob, numpy as np, io, contextlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/src/layered"); sys.path.insert(0, f"{ROOT}/src/kagome"); sys.path.insert(0, f"{ROOT}/src/kag")
from fitlib import Fitter
from layered_gain import rot_to_z, cell_triads, CI, NBR, global_triad
from cfg_terms import n_defects
T = float(sys.argv[1]); files = sorted(glob.glob(f"{ROOT}/data_kag/seeds/kag_L12_T{T:.4f}_s*.npy"), key=lambda f: int(f.split("_s")[-1][:-4]))
def loops_cellframe(X, Ucs, col):                                    # layered_gain2.loops_cellframe (verbatim logic)
    out = []
    for a in range(3):
        mask = col != a; ua = Ucs[:, a]; uc = Ucs[np.arange(len(X)), col]; E = uc - np.sum(uc * ua, 1, keepdims=True) * ua; E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9); W = np.cross(ua, E)
        chi = np.arctan2(np.sum(X * W, 1), np.sum(X * E, 1)); adj = {i: [j for j in NBR[i] if mask[j] and col[j] != col[i]] for i in np.where(mask)[0]}; seen = set()
        for i in adj:
            if i in seen: continue
            comp = []; st = [i]; ok = True
            while st:
                v = st.pop()
                if v in seen: continue
                seen.add(v); comp.append(v); ok &= len(adj[v]) == 2; st += [w for w in adj[v] if w not in seen]
            if ok and len(comp) >= 6: comp = np.array(comp); out.append((a, comp, np.angle(np.mean(np.exp(1j * chi[comp]))), len(comp)))
    return out
def rot_axis(v, ax, ang): c, s = np.cos(ang), np.sin(ang); return v * c + np.cross(ax, v) * s + ax * np.sum(ax * v, 1, keepdims=True) * (1 - c)
store = {}; desc = []
for fi, f in enumerate(files):
    PA, PB, PC = [], [], []; eps2 = []; psi = []; nd = []; nl = []
    for S in np.load(f).astype(float)[::4]:
        X = S.reshape(-1, 3); U = global_triad(S); col = np.argmax(X @ U.T, 1); nu_g = np.linalg.eigh(X.T @ X)[1][:, 0]
        RA = rot_to_z(U[col], np.tile(nu_g, (len(X), 1))); PA.append(np.einsum('nij,nj->ni', RA, X)); Uc, nuc = cell_triads(S, U, nu_g); Ucs = Uc[CI]; RB = rot_to_z(Ucs[np.arange(len(X)), col], nuc[CI]); PB.append(np.einsum('nij,nj->ni', RB, X))
        lp = loops_cellframe(X, Ucs, col); Xc = X.copy()
        for a, comp, chib, l in lp: Xc[comp] = rot_axis(Xc[comp], Ucs[comp, a], -chib)
        PC.append(np.einsum('nij,nj->ni', RB, Xc)); nl.append(len(lp)); eps2.append(np.mean(np.arcsin(np.clip(np.abs(X @ nu_g), 0, 1)) ** 2)); nd.append(n_defects(S, 3))
        c = S.reshape(4, 3, 4, 3, 3, 3).transpose(0, 2, 1, 3, 4, 5).reshape(-1, 27, 3); Q = np.einsum('cni,cnj->cij', c, c); V = np.linalg.eigh(Q)[1]; e1 = V[:, :, 2]; e2 = np.cross(V[:, :, 0], e1)
        ph = np.arctan2(np.einsum('cni,ci->cn', c, e2), np.einsum('cni,ci->cn', c, e1)); psi.append(np.abs(np.mean(np.exp(3j * ph), 1)).mean())
    seed = int(f.split("_s")[-1][:-4]); store[seed] = (np.concatenate(PA), np.concatenate(PB), np.concatenate(PC)); desc.append(dict(seed=seed, eps2=float(np.mean(eps2)), psi=float(np.mean(psi)), n_def=float(np.mean(nd)), loops=float(np.mean(nl)))); print("seed", seed, flush=True)
np.savez_compressed(f"{HERE}/data/layered_pools_T{T:.4f}.npz", **{f"{k}_{n}": store[k][i] for k in store for i, n in enumerate("ABC")})
def fit(arrs, L=6, order=131): return Fitter(L, order=order, alpha=1e-8).fit(np.concatenate(arrs)).nll
res = dict(T=T, desc=desc)
def abc(seeds, L=6): return [fit([store[s][i] for s in seeds], L) for i in range(3)]
res["all40"] = abc(list(store)); res["seeds1_6"] = abc([s for s in store if s <= 6]); res["seeds7_40"] = abc([s for s in store if s > 6]); res["seeds1_6_L8"] = abc([s for s in store if s <= 6], 8); res["all40_L8"] = abc(list(store), 8)
res["per_seed"] = {str(s): abc([s]) for s in store}                                        # per-seed pooled fits (25 snapshots)
# alternative quadrature: product Gauss-Legendre grid instead of Lebedev 131
from numpy.polynomial.legendre import leggauss
from fitlib import real_sh
def fit_prod(arrs, L=6, nt=180, nph=360):
    f = Fitter(L, order=59, alpha=1e-8); x, w = leggauss(nt); ph = np.arange(nph) * 2 * np.pi / nph; s = np.sqrt(1 - x ** 2); X = np.stack(np.broadcast_arrays(s[:, None] * np.cos(ph)[None], s[:, None] * np.sin(ph)[None], x[:, None] + 0 * ph[None]), -1).reshape(-1, 3)
    f.Yq, f.idx = real_sh(L, X); f.W = (w[:, None] * np.ones(nph) * 2 * np.pi / nph).ravel(); return f.fit(np.concatenate(arrs)).nll
res["all40_prodquad"] = [fit_prod([store[s][i] for s in store]) for i in range(3)]; res["all40_L6_leb131_check"] = res["all40"]
json.dump(res, open(f"{HERE}/data/layered_T{T:.4f}.json", "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k not in ("desc", "per_seed")}), flush=True)
