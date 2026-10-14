"""v3 2.1: moment-estimator re-measurement of trained v2 decoders (no retraining).  python3 moment_vae.py out.json run1.pt run2.pt ..."""
import sys, json, numpy as np, torch
from scipy.integrate import lebedev_rule
from scipy.special import eval_legendre as Pl
from mvae import sh
from mvae2 import cell_features
from eval_A import load, frame_of
from momlib import est_vec
torch.set_num_threads(1)
X, W = lebedev_rule(59); X = torch.tensor(X.T, dtype=torch.float64); W = torch.tensor(W, dtype=torch.float64); W = W * 4 * np.pi / W.sum()
YQ = sh(range(1, 7), X); LOGW = W.log()
def moments(lam):
    """lam [...,48] -> (<P2>, <P4>) of p(n) ∝ exp(lam.Y(n)) about the principal axis d of its l=2 part (Lebedev quadrature)."""
    lam = lam.double(); d = frame_of(lam)[..., :, 2]                              # [...,3]
    p = torch.softmax(lam @ YQ.T + LOGW, -1)                                      # [...,Q]
    c = d @ X.T                                                                   # [...,Q] = cos(theta) to the director
    P2 = 1.5 * c ** 2 - .5; P4 = (35 * c ** 4 - 30 * c ** 2 + 3) / 8
    return (p * P2).sum(-1), (p * P4).sum(-1)
@torch.no_grad()
def measure(path, nsamp=8):
    m, a = load(path); S = torch.tensor(np.load(f"{a['data']}/ll_L12_T{a['T']:.3f}_{a['st']}.npy")); nv = max(len(S) // 10, 10)
    F = cell_features(S[-nv:], a["R"]); mu, lv = m.encode(feat=F); torch.manual_seed(0)
    res = dict(run=path.split("/")[-1][:-3], T=a["T"], st=a["st"], R=a["R"])
    def run(zs):
        m2 = torch.stack([moments(m.decode_cells(z))[0] for z in zs]).mean(0); m4 = torch.stack([moments(m.decode_cells(z))[1] for z in zs]).mean(0)
        return m2.numpy(), m4.numpy()
    zs_all = [m.sample(mu, lv) for _ in range(nsamp)]
    # per-sample estimate: est on every (config, cell, sample) separately, then average (does not mix distribution shapes across samples)
    each = [est_vec(*[t.numpy() for t in moments(m.decode_cells(z))]) for z in zs_all]
    al = np.stack([e[0] for e in each]); s2 = np.stack([e[1] for e in each])
    res["samp_each"] = dict(a_loc=float(np.nanmean(al)), sigma2=float(np.nanmean(s2)), sigma2_median=float(np.nanmedian(s2)), frac_neg=float((s2 < 0).mean()))
    for name, zs in [("samp", zs_all), ("mean", [mu])]:
        m2, m4 = run(zs); al, s2 = est_vec(m2, m4)                                # per (config, cell)
        pa, ps = est_vec(np.array([m2.mean()]), np.array([m4.mean()]))             # pooled moments
        res[name] = dict(a_loc=float(np.nanmean(al)), sigma2=float(np.nanmean(s2)), a_loc_pooled=float(pa[0]), sigma2_pooled=float(ps[0]),
                         frac_valid=float(np.isfinite(al).mean()), m2=float(m2.mean()), m4=float(m4.mean()))
    return res
if __name__ == "__main__":
    out = [measure(p) for p in sys.argv[2:]]
    for r in out: print(r["run"], "samp-each a_loc %.3f s2 %.4f (neg %.2f) | samp-avg a_loc %.3f s2 %.4f | mean a_loc %.3f s2 %.4f" % (r["samp_each"]["a_loc"], r["samp_each"]["sigma2"], r["samp_each"]["frac_neg"], r["samp"]["a_loc"], r["samp"]["sigma2"], r["mean"]["a_loc"], r["mean"]["sigma2"]), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
