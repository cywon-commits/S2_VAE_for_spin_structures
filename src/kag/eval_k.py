"""Evaluation of kagome VAE runs on the validation seeds.  python3 eval_k.py out.json run.pt ...
- NLL (posterior-sample z), KL, per-seed s.e.m.; latent power per order l; order-truncation NLL (decoder lambda_l>L zeroed, no refit)
- decoder-distribution moments per (config, cell) distribution, then averaged: pooled over the 3 sublattices, frame from the distribution itself:
  nu = min eigenvector of <n n^T>, phase phi0 = arg<e^{3 i phi}>/3 ;  -ln m3, ln m6/ln m3, ln m9/ln m3, <eps^2>, |<e^{i phi}>|, |<e^{2 i phi}>|"""
import sys, json, numpy as np, torch
from scipy.integrate import lebedev_rule
from kmodel import KVAE, kag_features, kag_stats, NY, LMAX
from mvae import sh
from kdata import load_T, VAL_SEEDS
torch.set_num_threads(1)
X, W = lebedev_rule(131); XQ = torch.tensor(X.T, dtype=torch.float64); LW = torch.tensor(W * 4 * np.pi / W.sum(), dtype=torch.float64).log(); YQ = sh(range(1, LMAX + 1), XQ)
def decoder_moments(lam):
    """lam [...,3,80] -> dict of moments of the sublattice-pooled distribution (one value per leading index)."""
    p = torch.softmax(lam.double() @ YQ.T + LW, -1).mean(-2)                        # [..., Q]
    Qt = torch.einsum("...q,qi,qj->...ij", p, XQ, XQ); w, V = torch.linalg.eigh(Qt); nu = V[..., :, 0]; e1 = V[..., :, 2]; e2 = torch.cross(nu, e1, dim=-1)
    x = XQ @ e1.unsqueeze(-1); y = XQ @ e2.unsqueeze(-1); z = (XQ @ nu.unsqueeze(-1)).squeeze(-1)
    phi = torch.atan2(y.squeeze(-1), x.squeeze(-1)); phi0 = torch.angle((p * torch.exp(3j * phi)).sum(-1)) / 3; ph = phi - phi0.unsqueeze(-1)
    m = lambda k: (p * torch.cos(k * ph)).sum(-1)
    return dict(m3=m(3), m6=m(6), m9=m(9), eps2=(p * torch.asin(z.clamp(-1, 1)) ** 2).sum(-1), c1=torch.abs((p * torch.exp(1j * ph)).sum(-1)), c2=torch.abs((p * torch.exp(2j * ph)).sum(-1)))
def harmonic_ratios(d):
    m3, m6, m9 = d["m3"].mean().item(), d["m6"].mean().item(), d["m9"].mean().item()
    return dict(minus_ln_m3=-np.log(m3), ln_m6_over_ln_m3=np.log(m6) / np.log(m3), ln_m9_over_ln_m3=np.log(m9) / np.log(m3), eps2=d["eps2"].mean().item(), c1=d["c1"].mean().item(), c2=d["c2"].mean().item())
@torch.no_grad()
def evaluate_k(path, nsamp=4):
    ck = torch.load(path, weights_only=False); a = ck["args"]; R = a["R"]; m = KVAE(R=R, order=131); m.load_state_dict({k: v for k, v in ck["state"].items() if not k.startswith("logz.")}, strict=False); m.eval(); nsub = (12 // R) ** 2
    S = torch.tensor(load_T(a["T"], VAL_SEEDS)); F = kag_features(S, R); St = kag_stats(S, R); nseed = len(VAL_SEEDS); per = len(S) // nseed
    mu, lv = m.encode(F); kl = m.kl(mu, lv); torch.manual_seed(0); nll = torch.zeros(len(S)); mom = []; trunc = {L: torch.zeros(len(S)) for L in (1, 2, 3, 4, 6, 8)}
    for _ in range(nsamp):
        lam = m.decode(m.sample(mu, lv)); nll += m.nll_cfg(lam, St, nsub) / (3 * 144) / nsamp; mom.append(decoder_moments(lam))
        for L in trunc:
            lt = lam.clone(); lt[..., sum(2 * l + 1 for l in range(1, L + 1)):] = 0; trunc[L] += m.nll_cfg(lt, St, nsub) / (3 * 144) / nsamp
    seedmean = lambda v: v.reshape(nseed, per).mean(1)
    cat = {k: torch.cat([d[k].reshape(-1) for d in mom]) for k in mom[0]}
    pw = [float((mu[..., s:s + (2 * l + 1) * 4] ** 2).sum()) for l, s in zip(range(4), [0, 4, 16, 36])]
    return dict(run=path.split("/")[-1][:-3], T=a["T"], R=R, nll=nll.mean().item(), nll_se_seeds=seedmean(nll).std().item() / np.sqrt(nseed), kl=kl.mean().item(), kl_se_seeds=seedmean(kl).std().item() / np.sqrt(nseed),
                latent_power_fraction=[x / sum(pw) for x in pw], active_components=int(m.zmask.sum().item()), nll_truncated={str(L): v.mean().item() for L, v in trunc.items()}, decoder_moments=harmonic_ratios(cat))
if __name__ == "__main__":
    out = [evaluate_k(p) for p in sys.argv[2:]]
    for r in out:
        d = r["decoder_moments"]; print(r["run"], "NLL %.4f+-%.4f KL %.1f+-%.1f | latent power l0..3 %s | -ln m3 %.3f ln m6/ln m3 %.2f ln m9/ln m3 %.2f eps2 %.3f |e^i| %.3f |e^2i| %.3f | trunc %s" % (r["nll"], r["nll_se_seeds"], r["kl"], r["kl_se_seeds"], [round(x, 3) for x in r["latent_power_fraction"]], d["minus_ln_m3"], d["ln_m6_over_ln_m3"], d["ln_m9_over_ln_m3"], d["eps2"], d["c1"], d["c2"], {k: round(v, 3) for k, v in r["nll_truncated"].items()}), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
