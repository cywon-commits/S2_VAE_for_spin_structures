"""E3 C ring oracle: with the two-colour loops of kagome/loops.py, rotate each spin about the loop axis by minus the loop-mean angle chi_bar (per loop, sites are in 2 loops),
refit the multipole family (per sublattice, global frame) and compare NLL with the uncorrected fit; shuffled control assigns chi_bar to a random other loop of the same length.
python3 ring_oracle.py out.json T [nconf]"""
import sys, json, numpy as np
sys.path.insert(0, ".."); sys.path.insert(0, "../kagome")
from fitlib import Fitter
from kdata import load_T, VAL_SEEDS
from oracle_k import to_frame
from kagome_mc import NB
def build_nbrs(L):
    nb = np.zeros((L, L, 3, 4), int); idx = lambda x, y, s: ((x % L) * L + (y % L)) * 3 + s
    for x in range(L):
        for y in range(L):
            for s in range(3): nb[x, y, s] = [idx(x + dx, y + dy, t) for t, dx, dy in NB[s]]
    return nb.reshape(-1, 4)
def global_triad(X):
    V = np.linalg.eigh(X.T @ X)[1]; nu, e1 = V[:, 0], V[:, 2]; e2 = np.cross(nu, e1); phi = np.arctan2(X @ e2, X @ e1); th = np.angle(np.mean(np.exp(3j * phi))) / 3
    return np.array([np.cos(th + 2 * np.pi * k / 3) * e1 + np.sin(th + 2 * np.pi * k / 3) * e2 for k in range(3)])
def find_loops(X, NBR):
    U = global_triad(X); col = np.argmax(X @ U.T, 1); out = []
    for a in range(3):
        mask = col != a; ua = U[a]; E = U[col] - np.outer(U[col] @ ua, ua); E /= np.linalg.norm(E, axis=1, keepdims=True); W = np.cross(ua, E)
        chi = np.arctan2(np.sum(X * W, 1), np.sum(X * E, 1)); adj = {i: [j for j in NBR[i] if mask[j] and col[j] != col[i]] for i in np.where(mask)[0]}; seen = set()
        for i in adj:
            if i in seen: continue
            comp = []; stack = [i]; ok = True
            while stack:
                v = stack.pop()
                if v in seen: continue
                seen.add(v); comp.append(v)
                if len(adj[v]) != 2: ok = False
                stack += [w for w in adj[v] if w not in seen]
            if ok and len(comp) >= 6: out.append((np.array(comp), ua, np.angle(np.mean(np.exp(1j * chi[comp])))))
    return out
def rot(X, ax, ang):          # Rodrigues, rotate rows of X about unit ax by ang
    c, s = np.cos(ang), np.sin(ang); return X * c + np.cross(ax, X) * s + np.outer(X @ ax, ax) * (1 - c)
def correct(X, loops, chis):
    Y = X.copy()
    for (comp, ua, _), ch in zip(loops, chis): Y[comp] = rot(Y[comp], ua, -ch)
    return Y
if __name__ == "__main__":
    T = float(sys.argv[2]); NC = int(sys.argv[3]) if len(sys.argv) > 3 else 80; S = load_T(T, VAL_SEEDS); S = S[np.linspace(0, len(S) - 1, NC).astype(int)]
    NBR = build_nbrs(12); rng = np.random.default_rng(0); A = {"none": [], "ring": [], "shuffled": []}; cover = []; hexchi = []; allchi = []
    for cfg in S:
        X = cfg.reshape(-1, 3).astype(float); lp = find_loops(X, NBR); chis = [l[2] for l in lp]; lens = np.array([len(l[0]) for l in lp])
        cover.append(len(set(np.concatenate([l[0] for l in lp]).tolist())) / 432 if lp else 0)
        sh = np.array(chis).copy()
        for L in np.unique(lens): ii = np.where(lens == L)[0]; sh[ii] = rng.permutation(sh[ii])
        hexchi += [c ** 2 for c, n in zip(chis, lens) if n == 6]; allchi += [(n, c ** 2) for c, n in zip(chis, lens)]
        for k, ch in (("none", [0.0] * len(lp)), ("ring", chis), ("shuffled", list(sh))):
            A[k].append(correct(X, lp, ch).reshape(12, 12, 3, 3))
    res = dict(T=T, nconf=NC, site_coverage=float(np.mean(cover)), hex_chi2=float(np.mean(hexchi)), hex_chi2_over_sqrtT_ell=float(np.mean(hexchi) * 6 / np.sqrt(T)))
    for k, v in A.items():
        F = to_frame(np.stack(v)); res[k] = float(np.mean([Fitter(8, order=131, alpha=1e-8).fit(F[:, :, :, s].reshape(-1, 3)).nll for s in range(3)]))
    res["gain_ring"] = res["none"] - res["ring"]; res["gain_shuffled"] = res["none"] - res["shuffled"]
    json.dump(res, open(sys.argv[1], "w"), indent=1); print(res, flush=True)
