"""E3 B/C diagnostics: harmonic ratios of the DATA in cell frames.
 (a) data-own frames: per cell, nu = min eigenvector of the cell Q tensor, phi0 = arg<e^{3 i phi}>_cell/3 (as in kagome/local_frames.py), with a site-shuffled control (overfit);
 (b) decoder frames: per cell, the frame (nu, phi0) of the decoded (posterior-mean) pooled distribution of an up/down VAE run, applied to the data spins.
Pooled moments:  -ln m3, ln m6/ln m3, ln m9/ln m3, <eps^2>.   python3 cellframe_k.py out.json T R [run.pt]"""
import sys, json, numpy as np, torch
from kdata import load_T, VAL_SEEDS
def frames_from_sites(P):
    """P [cells, n, 3] -> frame columns (x',y',nu) [cells,3,3] from the cell's own Q tensor and cubic phase."""
    Q = np.einsum('cni,cnj->cij', P, P) / P.shape[1]; w, V = np.linalg.eigh(Q); nu = V[:, :, 0]; e1 = V[:, :, 2]; e2 = np.cross(nu, e1)
    x = np.einsum('cni,ci->cn', P, e1); y = np.einsum('cni,ci->cn', P, e2); phi0 = np.angle(np.mean(np.exp(3j * np.arctan2(y, x)), 1)) / 3
    xp = np.cos(phi0)[:, None] * e1 + np.sin(phi0)[:, None] * e2; yp = np.cross(nu, xp); return np.stack([xp, yp, nu], -1)
def cell_sites(S, R):
    C, L = S.shape[:2]; b = L // R; return S.reshape(C, R, b, R, b, 3, 3).transpose(0, 1, 3, 2, 4, 5, 6).reshape(C * R * R, b * b * 3, 3)
def pooled(P):
    P = P.reshape(-1, 3); z = np.clip(P[:, 2], -1, 1); phi = np.arctan2(P[:, 1], P[:, 0]); m = [np.mean(np.cos(k * phi)) for k in (3, 6, 9)]
    return dict(minus_ln_m3=float(-np.log(m[0])), ln_m6_over_ln_m3=float(np.log(m[1]) / np.log(m[0])), ln_m9_over_ln_m3=float(np.log(m[2]) / np.log(m[0])), eps2=float(np.mean(np.arcsin(z) ** 2)))
def data_own(T, R):
    S = load_T(T, VAL_SEEDS).astype(float); sites = cell_sites(S, R); F = frames_from_sites(sites); r = pooled(np.einsum('cni,cij->cnj', sites, F))
    rng = np.random.default_rng(0); sh = S.reshape(len(S), -1, 3).copy()
    for k in range(len(S)): sh[k] = sh[k][rng.permutation(sh.shape[1])]
    sh = sh.reshape(S.shape); ss = cell_sites(sh, R); rs = pooled(np.einsum('cni,cij->cnj', ss, frames_from_sites(ss))); return r, rs
def decoder_frames(T, R, path):
    from kmodel import kag_features
    from eval_ku import load
    from eval_k import XQ, YQ, LW
    m, a = load(path); S = torch.tensor(load_T(T, VAL_SEEDS)); F = kag_features(S, R)
    with torch.no_grad():
        mu, lv = m.encode(F); lam = m.decode(mu).double(); p = torch.softmax(lam @ YQ.T + LW, -1).mean(-2)
        Qt = torch.einsum("...q,qi,qj->...ij", p, XQ, XQ); w, V = torch.linalg.eigh(Qt); nu = V[..., :, 0]; e1 = V[..., :, 2]; e2 = torch.cross(nu, e1, dim=-1)
        x = XQ @ e1.unsqueeze(-1); y = XQ @ e2.unsqueeze(-1); phi = torch.atan2(y.squeeze(-1), x.squeeze(-1)); phi0 = torch.angle((p * torch.exp(3j * phi)).sum(-1)) / 3
        xp = torch.cos(phi0).unsqueeze(-1) * e1 + torch.sin(phi0).unsqueeze(-1) * e2; yp = torch.cross(nu, xp, dim=-1); Fr = torch.stack([xp, yp, nu], -1).reshape(-1, 3, 3).numpy()
    sites = cell_sites(S.numpy().astype(float), R); return pooled(np.einsum('cni,cij->cnj', sites, Fr))
if __name__ == "__main__":
    T = float(sys.argv[2]); R = int(sys.argv[3]); r, rs = data_own(T, R); out = dict(T=T, R=R, data_own_frames=r, data_own_frames_shuffled=rs)
    if len(sys.argv) > 4: out["decoder_frames"] = decoder_frames(T, R, sys.argv[4])
    json.dump(out, open(sys.argv[1], "w"), indent=1); f = lambda d: "-ln m3 %.3f ln m6/ln m3 %.2f ln m9/ln m3 %.2f eps2 %.3f" % (d["minus_ln_m3"], d["ln_m6_over_ln_m3"], d["ln_m9_over_ln_m3"], d["eps2"])
    print(f"T={T} R={R}: data own frames: {f(r)} | shuffled control: {f(rs)}" + (f" | decoder frames: {f(out['decoder_frames'])}" if "decoder_frames" in out else ""), flush=True)

def subset_ratios(T, R, path, thr=0.95):
    """Defect-free cells (local octupolar order |psi_cell| >= thr, computed from the data in the cell's own frame): data harmonic ratios vs the decoder's own ratios on the same cells."""
    from kmodel import kag_features
    from eval_ku import load
    from eval_k import decoder_moments
    S = load_T(T, VAL_SEEDS).astype(float); sites = cell_sites(S, R); F = frames_from_sites(sites); P = np.einsum('cni,cij->cnj', sites, F)
    phi = np.arctan2(P[:, :, 1], P[:, :, 0]); psi = np.abs(np.mean(np.exp(3j * phi), 1)); sel = psi >= thr
    mk = lambda k: np.mean(np.cos(k * phi[sel])); m3, m6, m9 = mk(3), mk(6), mk(9)
    data = dict(frac=float(sel.mean()), minus_ln_m3=float(-np.log(m3)), ln_m6_over_ln_m3=float(np.log(m6) / np.log(m3)), ln_m9_over_ln_m3=float(np.log(m9) / np.log(m3)))
    m, a = load(path); Ft = kag_features(torch.tensor(S), R)
    with torch.no_grad():
        mu, lv = m.encode(Ft); torch.manual_seed(0); acc = {k: [] for k in ("m3", "m6", "m9")}
        for _ in range(4):
            d = decoder_moments(m.decode(m.sample(mu, lv))); [acc[k].append(d[k].reshape(-1).numpy()) for k in acc]       # [configs*cells], same ordering as `sites`
    am = {k: np.mean(acc[k], 0)[sel].mean() for k in acc}
    dec = dict(minus_ln_m3=float(-np.log(am["m3"])), ln_m6_over_ln_m3=float(np.log(am["m6"]) / np.log(am["m3"])), ln_m9_over_ln_m3=float(np.log(am["m9"]) / np.log(am["m3"])))
    return data, dec
