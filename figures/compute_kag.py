"""Recomputation of the chat-side kagome analyses on the repository seeds (data_kag/seeds, 40 seeds x 100 snapshots per T; every 4th snapshot, as in the chat scripts -> 1000 per T).
Sources: kagome/local_frames.py, defect_check.py, zeromode_check.py, block_defects.py, loops.py (zip mvae_small_calcs).  python3 compute_kag.py T   -> figures/data/kag_T{T}.json
Standard errors: over the 40 seeds (per-seed statistics)."""
import sys, os, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/src/kag"); sys.path.insert(0, f"{ROOT}/src/layered")
from loops import build_nbrs, global_triad                        # shim: helpers of kagome/loops.py without the module-level analysis loop
from cfg_terms import block_frames
from scipy.spatial.transform import Rotation as Rot
T = float(sys.argv[1]); files = sorted(glob.glob(f"{ROOT}/data_kag/seeds/kag_L12_T{T:.4f}_s*.npy"), key=lambda f: int(f.split("_s")[-1][:-4])); NBR = build_nbrs(12); BINS = [(0.95, 1.01), (0.85, 0.95), (0.6, 0.85), (0.0, 0.6)]
def cells(S, b):                                                   # defect_check.cells
    L = S.shape[0]; c = S.reshape(L // b, b, L // b, b, 3, 3).transpose(0, 2, 1, 3, 4, 5).reshape(-1, 3 * b * b, 3); Q = np.einsum('cni,cnj->cij', c, c); V = np.linalg.eigh(Q)[1]; nu = V[:, :, 0]; e1 = V[:, :, 2]; e2 = np.cross(nu, e1)
    z = np.einsum('cni,ci->cn', c, nu); phi = np.arctan2(np.einsum('cni,ci->cn', c, e2), np.einsum('cni,ci->cn', c, e1)); psi = np.mean(np.exp(3j * phi), 1); return z, phi - np.angle(psi)[:, None] / 3, np.abs(psi)
def tri_energy_cells(S, b=3):                                       # zeromode_check.py
    A, B, C = S[:, :, 0], S[:, :, 1], S[:, :, 2]; up = A + B + C; dn = A + np.roll(B, 1, axis=0) + np.roll(C, 1, axis=1); e = (np.sum(up ** 2, -1) + np.sum(dn ** 2, -1)) / 2; L = S.shape[0]; return e.reshape(L // b, b, L // b, b).mean(axis=(1, 3)).ravel()
def classify_blocks(S, b):                                          # block_defects.py
    C = block_frames(S, b); r = lambda X, dx, dy: np.roll(X, (-dx, -dy), axis=(0, 1)); loop = [C, r(C, 1, 0), r(C, 1, 1), r(C, 0, 1), C]; q = C[:, :, 0, :]
    for X in loop[1:]: idx = np.argmax(np.einsum('xyci,xyi->xyc', X, q), -1); q = np.take_along_axis(X, idx[..., None, None], 2)[:, :, 0, :]
    k = idx // 4; sg = (idx // 2) % 2; mn = idx % 2
    return np.where((k == 0) & (sg == 0) & (mn == 0), 'none', np.where((k == 0) & (sg == 0), 'Z2', np.where(sg == 0, 'C3', 'C2')))
def loops_cfg(S):                                                   # loops.py: two-colour loops, chi_bar and in-loop rms
    X = S.reshape(-1, 3); U = global_triad(S); col = np.argmax(X @ U.T, 1); out = []
    for a in range(3):
        mask = col != a; ua = U[a]; E = U[col] - np.outer(U[col] @ ua, ua); E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9); W = np.cross(ua, E); chi = np.arctan2(np.sum(X * W, 1), np.sum(X * E, 1))
        adj = {i: [j for j in NBR[i] if mask[j] and col[j] != col[i]] for i in np.where(mask)[0]}; seen = set()
        for i in adj:
            if i in seen: continue
            comp = []; st = [i]; ok = True
            while st:
                v = st.pop()
                if v in seen: continue
                seen.add(v); comp.append(v); ok &= len(adj[v]) == 2; st += [w for w in adj[v] if w not in seen]
            if ok and len(comp) >= 6: c = chi[comp]; m = np.angle(np.mean(np.exp(1j * c))); out.append((len(comp), m, np.sqrt(np.mean(np.angle(np.exp(1j * (c - m))) ** 2))))
    return out
def mom(Z, P):
    e2 = np.mean(np.arcsin(np.clip(Z, -1, 1)) ** 2); m3, m6, m9 = [np.mean(np.cos(k * P)) for k in (3, 6, 9)]; return e2, -np.log(m3), np.log(m6) / np.log(m3), np.log(m9) / np.log(m3)
per_seed = []; loops_all = []; hexchi = []; defects = {b: dict(Z2=0, C3=0, C2=0, n=0) for b in (2, 3)}; zm = {str(bb): [] for bb in BINS}
for f in files:
    snaps = np.load(f).astype(float)[::4]; Zg, Pg, Zc, Pc, Ac = [], [], [], [], []; ps = {}
    for S in snaps:
        z, p, A = cells(S, 12); Zg.append(z.ravel()); Pg.append(p.ravel()); z, p, A = cells(S, 3); Zc.append(z); Pc.append(p); Ac.append(A)
        e = tri_energy_cells(S, 3); eps2 = np.mean(np.arcsin(np.clip(z, -1, 1)) ** 2, 1)
        for bb in BINS: s = (A >= bb[0]) & (A < bb[1]); zm[str(bb)] += list(zip(e[s], eps2[s]))
        for b in (2, 3): c = classify_blocks(S, b); [defects[b].__setitem__(k, defects[b][k] + int((c == k).sum())) for k in ("Z2", "C3", "C2")]; defects[b]["n"] += c.size
        if T <= 0.0041:
            lp = loops_cfg(S); loops_all += lp
    Zg = np.concatenate(Zg); Pg = np.concatenate(Pg); Zc = np.concatenate(Zc); Pc = np.concatenate(Pc); Ac = np.concatenate(Ac)
    rec = dict(glob=mom(Zg, Pg), cell3=mom(Zc.ravel(), Pc.ravel()))
    for bb in BINS:
        s = (Ac >= bb[0]) & (Ac < bb[1]); rec[str(bb)] = dict(frac=float(s.mean()), mom=mom(Zc[s].ravel(), Pc[s].ravel()) if s.sum() >= 5 else None)
    per_seed.append(rec); print("seed", f.split("_s")[-1], flush=True)
def agg(vals): v = np.array(vals, float); return dict(mean=float(np.nanmean(v)), se=float(np.nanstd(v, ddof=1) / np.sqrt(np.sum(~np.isnan(v)))), n=int(np.sum(~np.isnan(v))))
out = dict(T=T, n_seeds=len(files), glob={k: agg([r["glob"][i] for r in per_seed]) for i, k in enumerate(("eps2", "minus_ln_m3", "r6", "r9"))}, cell3={k: agg([r["cell3"][i] for r in per_seed]) for i, k in enumerate(("eps2", "minus_ln_m3", "r6", "r9"))})
out["bins"] = {str(bb): dict(frac=agg([r[str(bb)]["frac"] for r in per_seed]), **{k: agg([r[str(bb)]["mom"][i] if r[str(bb)]["mom"] else np.nan for r in per_seed]) for i, k in enumerate(("eps2", "minus_ln_m3", "r6", "r9"))}) for bb in BINS}
out["zeromode"] = {k: dict(n=len(v), tri_over_T=float(np.mean([x[0] for x in v]) / T) if v else None, eps2=float(np.mean([x[1] for x in v])) if v else None) for k, v in zm.items()}
out["defects"] = {str(b): {k: defects[b][k] / defects[b]["n"] for k in ("Z2", "C3", "C2")} | dict(per_block_n=defects[b]["n"], per_config={k: defects[b][k] / (len(files) * 25) for k in ("Z2", "C3", "C2")}) for b in (2, 3)}
if loops_all:
    a = np.array(loops_all); out["loops"] = dict(length=a[:, 0].tolist(), chi=a[:, 1].tolist(), rms=a[:, 2].tolist(), n_config=len(files) * 25)
json.dump(out, open(f"{HERE}/data/kag_T{T:.4f}.json", "w"), default=float); print("done", T)
