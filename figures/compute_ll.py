"""Recomputation of the chat-side nematic analyses on the repository data (data/ll_L12_T*_o.npy, 2000 snapshots per T, ordered side only).
Outputs figures/data/ll_derived.json.  Sources of the functions: moment_est.py, budget.py/budget2.py, cell_gain.py, amp_gain.py, mix_check.py (zip mvae_small_calcs; paths changed to the repository).
Uncertainties: s.e. over 5 blocks of configurations (blocks of 20 configs for the cell analyses, of 400 for the moments)."""
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, f"{ROOT}/src")
from numpy.polynomial.legendre import leggauss
from scipy.optimize import fsolve
from scipy.special import eval_legendre as Pl
from scipy.interpolate import interp1d
from fitlib import Fitter
x, w = leggauss(400); P2 = lambda c: 1.5 * c ** 2 - .5
def M(a, l): p = np.exp(a * Pl(2, x)); return np.sum(w * p * Pl(l, x)) / np.sum(w * p)
def est(m2, m4):
    f = lambda v: [np.exp(-1.5 * v[1]) * M(v[0], 2) - m2, np.exp(-5 * v[1]) * M(v[0], 4) - m4]; a, s2 = fsolve(f, [4.0, 0.03]); return float(a), float(s2)
def Idir(a2):
    p = np.exp(a2 * P2(x)); p /= np.sum(w * p); return 4.5 * a2 ** 2 * np.sum(w * p * x ** 2 * (1 - x ** 2))
TS = [0.80, 0.95, 1.05, 1.10]; out = {"T": TS}
def load(T): return np.load(f"{ROOT}/data/ll_L12_T{T:.3f}_o.npy").astype(float)
# ---------------- moments (global Q frame, pooled)
mom = {}
for T in TS:
    S = load(T); res = []
    for blk in np.array_split(np.arange(len(S)), 5):
        Sb = S[blk]; Q = np.einsum('cxyzi,cxyzj->cij', Sb, Sb) / 12 ** 3; d = np.linalg.eigh(Q)[1][:, :, -1]; c = np.abs(np.einsum('cxyzi,ci->cxyz', Sb, d)).ravel()
        m2 = np.mean(Pl(2, c)); m4 = np.mean(Pl(4, c)); a, s2 = est(m2, m4); res.append((m2, a, s2, 6 * m2 / T))
    r = np.array(res); mom[str(T)] = dict(S=r[:, 0].mean(), S_se=r[:, 0].std(ddof=1) / np.sqrt(5), a_loc=r[:, 1].mean(), a_loc_se=r[:, 1].std(ddof=1) / np.sqrt(5), sigma2_site=r[:, 2].mean(), sigma2_site_se=r[:, 2].std(ddof=1) / np.sqrt(5), a_mf=r[:, 3].mean(), a_mf_se=r[:, 3].std(ddof=1) / np.sqrt(5))
    print("moments", T, mom[str(T)], flush=True)
out["moments"] = mom
# ---------------- cell spread (budget.py) and cell-frame NLL gains (cell_gain.py), first 100 configs as in the chat scripts, 5 blocks of 20
a2o = {0.80: 5.51, 0.95: 3.97, 1.05: 2.92, 1.10: 2.16}                                  # global oracle a2 (REPORT/oracle.json); recomputed below from oracle.json
oj = json.load(open(f"{ROOT}/results/oracle.json")); a2o = {round(r["T"], 2): r["a2"] for r in oj if round(r["T"], 2) in a2o}
rng = np.random.default_rng(0); N = 12 ** 3
def frame_dir(S): Q = np.einsum('cxyzi,cxyzj->cij', S, S) / N; return np.linalg.eigh(Q)[1][:, :, -1]
def cell_spread(S, dg, b):
    c, L = S.shape[0], S.shape[1]; Sb = S.reshape(c, L // b, b, L // b, b, L // b, b, 3); Qb = np.einsum('cxaybzdi,cxaybzdj->cxyzij', Sb, Sb); db = np.linalg.eigh(Qb)[1][..., -1]
    cos = np.abs(np.einsum('cxyzi,ci->cxyz', db, dg)); return np.mean(np.arccos(np.clip(cos, 0, 1)) ** 2)
def shuffle(S):
    Ssh = S.reshape(len(S), -1, 3).copy()
    for k in range(len(S)): Ssh[k] = Ssh[k][rng.permutation(N)]
    return Ssh.reshape(S.shape)
def frames_global(S): Q = np.einsum('cxyzi,cxyzj->cij', S, S); return np.linalg.eigh(Q)[1][:, :, [1, 0, 2]]
def rotate_cells(S, b):
    c, L = S.shape[0], S.shape[1]; Sb = S.reshape(c, L // b, b, L // b, b, L // b, b, 3); Qb = np.einsum('cxaybzdi,cxaybzdj->cxyzij', Sb, Sb); V = np.linalg.eigh(Qb)[1][..., [1, 0, 2]]
    return np.einsum('cxaybzdi,cxyzij->cxaybzdj', Sb, V).reshape(-1, 3)
def nll(P): return Fitter(4, order=59, alpha=1e-8).fit(P).nll
cells = {}
for T in TS:
    S0 = load(T)[:100]; I = Idir(a2o[T]); rec = {R: [] for R in (2, 3, 4)}
    for blk in np.array_split(np.arange(100), 5):
        S = S0[blk]; dg = frame_dir(S); Ssh = shuffle(S); base = nll(np.einsum('cxyzi,cij->cxyzj', S, frames_global(S)).reshape(-1, 3)); base_s = nll(np.einsum('cxyzi,cij->cxyzj', Ssh, frames_global(Ssh)).reshape(-1, 3))
        for R, b in [(2, 6), (3, 4), (4, 3)]:
            s2 = max(cell_spread(S, dg, b) - cell_spread(Ssh, dg, b), 0.0); raw = base - nll(rotate_cells(S, b)); ov = base_s - nll(rotate_cells(Ssh, b)); rec[R].append((s2, raw, ov, raw - ov))
    cells[str(T)] = {}
    for R, b in [(2, 6), (3, 4), (4, 3)]:
        r = np.array(rec[R]); n = b ** 3; m = r.mean(0); se = r.std(0, ddof=1) / np.sqrt(5)
        cells[str(T)][str(R)] = dict(n=n, sigma2_cell=m[0], sigma2_cell_se=se[0], gain_raw=m[1], gain_overfit=m[2], gain_net=m[3], gain_net_se=se[3], I=I, G=n * I * m[0] / 2, G_se=n * I * se[0] / 2)
    print("cells", T, {R: (round(v['sigma2_cell'], 4), round(v['gain_net'], 4)) for R, v in cells[str(T)].items()}, flush=True)
out["cells"] = cells; out["a2_oracle"] = {str(k): v for k, v in a2o.items()}
# ---------------- amplitude channel (amp_gain.py)
xx, ww = leggauss(300); P2x = 1.5 * xx ** 2 - .5; A = np.linspace(-2, 40, 4000); Z = np.array([np.sum(ww * np.exp(a * P2x)) for a in A]); M2 = np.array([np.sum(ww * P2x * np.exp(a * P2x)) for a in A]) / Z
a_of_m = interp1d(M2, A, bounds_error=False, fill_value=(A[0], A[-1])); logZ = interp1d(A, np.log(Z))
def cell_m2(S, b):
    c, L = S.shape[0], S.shape[1]; Sb = S.reshape(c, L // b, b, L // b, b, L // b, b, 3); Qb = np.einsum('cxaybzdi,cxaybzdj->cxyzij', Sb, Sb); d = np.linalg.eigh(Qb)[1][..., -1]
    cs = np.einsum('cxaybzdi,cxyzi->cxyzabd', Sb, d); return (1.5 * cs ** 2 - .5).mean(axis=(-3, -2, -1)).ravel()
def amp_gain(m2c): ap = float(a_of_m(m2c.mean())); pooled = -(ap * m2c.mean() - logZ(ap)); ac = a_of_m(m2c); per = -np.mean(ac * m2c - logZ(ac)); return pooled - per
amp = {}
for T in (0.95, 1.05):
    S0 = load(T)[:100]; amp[str(T)] = {}
    for R, b in [(2, 6), (3, 4), (4, 3)]:
        v = []
        for blk in np.array_split(np.arange(100), 5):
            S = S0[blk]; v.append(amp_gain(cell_m2(S, b)) - amp_gain(cell_m2(shuffle(S), b)))
        v = np.array(v); n = b ** 3; amp[str(T)][str(R)] = dict(gain=v.mean(), gain_se=v.std(ddof=1) / np.sqrt(5), Ga=n * v.mean(), n=n)
    print("amp", T, amp[str(T)], flush=True)
out["amp"] = amp
# ---------------- synthetic validation of the moment estimator (moment_est.py) and mixing bias (mix_check.py)
from samplers import sample_bingham
rg = np.random.default_rng(3); Ns = 400000; val = []
for a2 in (2.0, 4.0, 8.0, 15.0):
    for s2 in (0.01, 0.04):
        base = sample_bingham(np.diag([0, 0, 1.5 * a2]), Ns, rg); tx, ty = rg.normal(scale=np.sqrt(s2 / 2), size=(2, Ns)); th = np.sqrt(tx ** 2 + ty ** 2); ph = np.arctan2(ty, tx)
        D = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1); z = np.array([0, 0, 1.]); ax = np.cross(z, D); s = np.linalg.norm(ax, axis=1); c = D[:, 2]; ax = ax / np.maximum(s, 1e-12)[:, None]
        Sm = base * c[:, None] + np.cross(ax, base) * s[:, None] + ax * (np.sum(ax * base, 1) * (1 - c))[:, None]; m2 = np.mean(Pl(2, Sm[:, 2])); m4 = np.mean(Pl(4, Sm[:, 2])); a, e = est(m2, m4)
        # a4 estimator (REQUEST: sigma^2 = -(35/18) a4/a2^2): a4 from the exact smeared log-density via the heat-kernel formula is only valid for small a2; here use the moment-based a4 proxy (see report)
        val.append(dict(a2=a2, s2_true=s2, a_loc=a, s2_est=e))
out["validation"] = val
def moms(a, s2): return np.exp(-1.5 * s2) * M(a, 2), np.exp(-5 * s2) * M(a, 4)
mix = []; rm = np.random.default_rng(0)
for a0 in (4.0, 3.0):
    for s2 in (0.03, 0.01):
        for r in (0.0, 0.05, 0.10, 0.20, 0.30):
            Aa = a0 * (1 + r * rm.normal(size=400)); ms = np.array([moms(a, s2) for a in Aa]); per = np.mean([est(*m)[1] for m in ms]); avg = est(*ms.mean(0))[1]; mix.append(dict(a0=a0, s2=s2, r=r, per_sample=per, avg_moment=avg))
out["mix"] = mix
json.dump(out, open(f"{HERE}/data/ll_derived.json", "w"), indent=1, default=float); print("done")
