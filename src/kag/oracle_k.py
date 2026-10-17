"""Global oracle on the validation seeds: rotate each snapshot into its global frame (nu = plane normal = min eigenvector of Q, phase phi0 = arg<e^{3 i phi}>/3)
and fit the unrestricted multipole family (fitlib.Fitter, L=8): pooled over sublattices ("global oracle") and per sublattice.  python3 oracle_k.py out.json T ..."""
import sys, json, numpy as np
sys.path.insert(0, ".."); from fitlib import Fitter
from kdata import load_T, VAL_SEEDS
def to_frame(S):
    X = S.reshape(len(S), -1, 3).astype(float); Q = np.einsum('cni,cnj->cij', X, X) / X.shape[1]; w, V = np.linalg.eigh(Q)
    nu = V[:, :, 0]; e1 = V[:, :, 2]; e2 = np.cross(nu, e1)
    x = np.einsum('cni,ci->cn', X, e1); y = np.einsum('cni,ci->cn', X, e2); phi0 = np.angle(np.mean(np.exp(3j * np.arctan2(y, x)), 1)) / 3
    xp = np.cos(phi0)[:, None] * e1 + np.sin(phi0)[:, None] * e2; yp = np.cross(nu, xp); Fr = np.stack([xp, yp, nu], -1)     # columns of the frame
    return np.einsum('cxysi,cij->cxysj', S.astype(float), Fr)
def moments(P):
    z = np.clip(P[:, 2], -1, 1); phi = np.arctan2(P[:, 1], P[:, 0]); m = [np.mean(np.cos(k * phi)) for k in (3, 6, 9)]
    return dict(minus_ln_m3=-np.log(m[0]), ln_m6_over_ln_m3=np.log(m[1]) / np.log(m[0]), ln_m9_over_ln_m3=np.log(m[2]) / np.log(m[0]), eps2=float(np.mean(np.arcsin(z) ** 2)))
if __name__ == "__main__":
    out = []
    for T in map(float, sys.argv[2:]):
        S = load_T(T, VAL_SEEDS); F = to_frame(S); P = F.reshape(-1, 3)
        fits = {L: Fitter(L, order=131 if L == 8 else 59, alpha=1e-8).fit(P) for L in (2, 3, 6, 8)}
        sub = [Fitter(8, order=131, alpha=1e-8).fit(F[:, :, :, s].reshape(-1, 3)).nll for s in range(3)]
        r = dict(T=T, nll_global=fits[8].nll, nll_by_L={str(L): f.nll for L, f in fits.items()}, nll_sublattice_resolved=float(np.mean(sub)), **moments(P)); out.append(r); print(r, flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
