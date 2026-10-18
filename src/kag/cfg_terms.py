"""E3 v2 C inputs: per validation configuration (i) ring-oracle gain g_ring (nats per site, in-sample multipole fits per sublattice pooled over the validation configs of that T:
g = mean_i [log p_ring(corrected n_i) - log p_none(n_i)], correction as in ring_oracle.py) and (ii) number of 3x3-block phase defects n_def (kagome/block_defects.py rule).
python3 cfg_terms.py T   ->  results/kag/cfgterms_T{T}.json"""
import sys, json, numpy as np
sys.path.insert(0, ".."); sys.path.insert(0, "../kagome")
from scipy.spatial.transform import Rotation as Rot
from fitlib import Fitter
from kdata import load_T, VAL_SEEDS
from oracle_k import to_frame
from ring_oracle import build_nbrs, find_loops, correct
def block_frames(S, b):
    L = S.shape[0]; c = S.reshape(L // b, b, L // b, b, 3, 3).transpose(0, 2, 1, 3, 4, 5).reshape(L // b, L // b, -1, 3)
    Q = np.einsum('xyni,xynj->xyij', c, c); V = np.linalg.eigh(Q)[1]; nu = V[..., 0]; e1 = V[..., 2]; e2 = np.cross(nu, e1)
    phi = np.arctan2(np.einsum('xyni,xyi->xyn', c, e2), np.einsum('xyni,xyi->xyn', c, e1)); psi = np.mean(np.exp(3j * phi), -1); th = np.angle(psi) / 3; Qs = []
    for k in range(3):
        a = th + 2 * np.pi * k / 3; u = np.cos(a)[..., None] * e1 + np.sin(a)[..., None] * e2
        for sg in (1, -1):
            e3 = sg * nu; M = np.stack([u, np.cross(e3, u), e3], -1); q = Rot.from_matrix(M.reshape(-1, 3, 3)).as_quat().reshape(M.shape[:-2] + (4,)); Qs += [q, -q]
    return np.stack(Qs, -2)
def n_defects(S, b=3):
    C = block_frames(S, b); r = lambda X, dx, dy: np.roll(X, (-dx, -dy), axis=(0, 1)); loop = [C, r(C, 1, 0), r(C, 1, 1), r(C, 0, 1), C]; q = C[:, :, 0, :]
    for X in loop[1:]:
        idx = np.argmax(np.einsum('xyci,xyi->xyc', X, q), -1); q = np.take_along_axis(X, idx[..., None, None], 2)[:, :, 0, :]
    k = idx // 4; sg = (idx // 2) % 2; mn = idx % 2; none = (k == 0) & (sg == 0) & (mn == 0); return int((~none).sum())
if __name__ == "__main__":
    T = float(sys.argv[1]); S = load_T(T, VAL_SEEDS).astype(float); NBR = build_nbrs(12); orig, corr, ndef = [], [], []
    for cfg in S:
        X = cfg.reshape(-1, 3); lp = find_loops(X, NBR); orig.append(cfg); corr.append(correct(X, lp, [l[2] for l in lp]).reshape(12, 12, 3, 3)); ndef.append(n_defects(cfg))
    Fo = to_frame(np.stack(orig)); Fc = to_frame(np.stack(corr)); g = np.zeros(len(S))
    def nll_by_cfg(F, fits):          # per-config mean nll per site under the per-sublattice fits
        out = np.zeros(len(F))
        for s in range(3):
            f = fits[s]; Z = np.log(np.sum(f.W * np.exp(f.Yq @ f.lam)))
            for c in range(len(F)): out[c] += (f.E(F[c, :, :, s].reshape(-1, 3)).mean() + Z) / 3
        return out
    fo = [Fitter(8, order=131, alpha=1e-8).fit(Fo[:, :, :, s].reshape(-1, 3)) for s in range(3)]; fc = [Fitter(8, order=131, alpha=1e-8).fit(Fc[:, :, :, s].reshape(-1, 3)) for s in range(3)]
    g = nll_by_cfg(Fo, fo) - nll_by_cfg(Fc, fc)
    json.dump(dict(T=T, g_ring=g.tolist(), n_def=ndef), open(f"../../results/kag/cfgterms_T{T:.4f}.json", "w")); print(T, "g_ring mean", g.mean(), "n_def mean", np.mean(ndef), flush=True)
