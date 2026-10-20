"""REQUEST_FIGURES_v2 C-4: -ln<cos3phi> and <eps^2> in the 3x3-cell frames (27 sites), excluding cells that touch a defective 2x2 block loop (any of Z2/C3/C2 from the block_defects.py rule).
python3 compute_kag_nodef.py T -> figures/data/kag_nodef_T{T}.json  (40 seeds, every 4th snapshot; s.e. over seeds)"""
import sys, os, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/src/kag"); sys.path.insert(0, f"{ROOT}/src/kagome")
from cfg_terms import block_frames
T = float(sys.argv[1]); files = sorted(glob.glob(f"{ROOT}/data_kag/seeds/kag_L12_T{T:.4f}_s*.npy"), key=lambda f: int(f.split("_s")[-1][:-4]))
def loop_classes(S, b=3):
    C = block_frames(S, b); r = lambda X, dx, dy: np.roll(X, (-dx, -dy), axis=(0, 1)); loop = [C, r(C, 1, 0), r(C, 1, 1), r(C, 0, 1), C]; q = C[:, :, 0, :]
    for X in loop[1:]: idx = np.argmax(np.einsum('xyci,xyi->xyc', X, q), -1); q = np.take_along_axis(X, idx[..., None, None], 2)[:, :, 0, :]
    k = idx // 4; sg = (idx // 2) % 2; mn = idx % 2; return ~((k == 0) & (sg == 0) & (mn == 0))            # True where the loop starting at block (x,y) is defective
def cells(S, b=3):
    L = S.shape[0]; c = S.reshape(L // b, b, L // b, b, 3, 3).transpose(0, 2, 1, 3, 4, 5).reshape(-1, 3 * b * b, 3); Q = np.einsum('cni,cnj->cij', c, c); V = np.linalg.eigh(Q)[1]; nu = V[:, :, 0]; e1 = V[:, :, 2]; e2 = np.cross(nu, e1)
    z = np.einsum('cni,ci->cn', c, nu); phi = np.arctan2(np.einsum('cni,ci->cn', c, e2), np.einsum('cni,ci->cn', c, e1)); psi = np.mean(np.exp(3j * phi), 1); return z, phi - np.angle(psi)[:, None] / 3
per = []
for f in files:
    Zs, Ps, Za, Pa = [], [], [], []; nex = 0; ncell = 0
    for S in np.load(f).astype(float)[::4]:
        d = loop_classes(S); bad = np.zeros((4, 4), bool)
        for x, y in zip(*np.where(d)):
            for dx in (0, 1):
                for dy in (0, 1): bad[(x + dx) % 4, (y + dy) % 4] = True
        z, p = cells(S); keep = ~bad.reshape(-1); Zs.append(z[keep]); Ps.append(p[keep]); Za.append(z); Pa.append(p); nex += (~keep).sum(); ncell += len(keep)
    mom = lambda Z, P: (float(np.mean(np.arcsin(np.clip(Z, -1, 1)) ** 2)), float(-np.log(np.mean(np.cos(3 * P)))))
    Zs = np.concatenate(Zs); Ps = np.concatenate(Ps); Za = np.concatenate(Za); Pa = np.concatenate(Pa); per.append(dict(nodef=mom(Zs, Ps), all=mom(Za, Pa), frac_excluded=nex / ncell))
agg = lambda v: dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1) / np.sqrt(len(v))))
out = dict(T=T, n_seeds=len(files), nodef_eps2=agg([p["nodef"][0] for p in per]), nodef_minus_ln_m3=agg([p["nodef"][1] for p in per]), all_eps2=agg([p["all"][0] for p in per]), all_minus_ln_m3=agg([p["all"][1] for p in per]), frac_excluded=agg([p["frac_excluded"] for p in per]))
json.dump(out, open(f"{HERE}/data/kag_nodef_T{T:.4f}.json", "w"), indent=1); print(out)
