"""v2 stage-A convergence criteria + decoder-frame oracle.   python3 eval_A.py run.pt out.json"""
import sys, json, numpy as np, torch
from e3nn import o3
from mvae import sh, LAM_IRREPS
from mvae2 import MVAE2, compute_stats, cell_features
from fitlib import Fitter
torch.set_num_threads(1)
SL = [(sum(2 * k + 1 for k in range(1, l)), sum(2 * k + 1 for k in range(1, l + 1))) for l in range(1, 7)]

def load(path):
    ck = torch.load(path, weights_only=False); a = ck["args"]; m = MVAE2(L=12, R=a["R"]); m.load_state_dict(ck["state"]); m.eval(); return m, a

_G = torch.randn(14, 3, dtype=torch.float64); _G = _G / _G.norm(dim=1, keepdim=True)
_Y2G = sh([2], _G); _SYM = torch.stack([_G[:, 0] ** 2, _G[:, 1] ** 2, _G[:, 2] ** 2, 2 * _G[:, 0] * _G[:, 1], 2 * _G[:, 0] * _G[:, 2], 2 * _G[:, 1] * _G[:, 2]], 1)
_PINV = torch.linalg.pinv(_SYM)                         # f(n)=n^T A n on 14 directions -> 6 symmetric-matrix entries
def frame_of(lam):
    """lam [...,48] -> orthonormal frame V [...,3,3] (columns x,y,z=director) from the l=2 part (principal axes of its quadratic form)."""
    l2 = lam[..., SL[1][0]:SL[1][1]].double(); f = l2 @ _Y2G.T                            # [...,14]
    c = f @ _PINV.T; A = torch.zeros(*c.shape[:-1], 3, 3, dtype=torch.float64)
    A[..., 0, 0], A[..., 1, 1], A[..., 2, 2] = c[..., 0], c[..., 1], c[..., 2]
    A[..., 0, 1] = A[..., 1, 0] = c[..., 3]; A[..., 0, 2] = A[..., 2, 0] = c[..., 4]; A[..., 1, 2] = A[..., 2, 1] = c[..., 5]
    w, V = torch.linalg.eigh(A); return V[..., [1, 0, 2]]
def a_l(lam):
    """lam [...,48] -> a_2, a_4, a_6 = <lam_l, Y_l(d)>, d = director of the l=2 part."""
    d = frame_of(lam)[..., :, 2]; out = []
    for l in (2, 4, 6): out.append((lam[..., SL[l - 1][0]:SL[l - 1][1]].double() * sh([l], d)).sum(-1))
    return out
def power_ratio(lam):
    P = [(lam[..., s:e] ** 2).sum(-1) for s, e in SL]; return sum(P[i] for i in (0, 2, 4)), sum(P[i] for i in (1, 3, 5))

@torch.no_grad()
def elbo_terms(m, F, St, nc, eps_mu, eps_lv, nsamp=64, seed=0):
    """MC ELBO per config (nats) with common random numbers; q = N(mu, exp(lv)) per channel."""
    g = torch.Generator().manual_seed(seed); tot = 0.
    kl = m.kl(eps_mu, eps_lv); lvx = m.expand_logvar(eps_lv)
    for _ in range(nsamp):
        z = eps_mu + torch.randn(eps_mu.shape, generator=g) * torch.exp(0.5 * lvx); lam = m.decode_cells(z)
        tot = tot + (-(lam * St).sum(-1) + nc * m.logz(lam)).sum(-1) / nsamp
    return -tot - kl, tot, kl

def amort_gap(m, F, St, nc, steps=600):
    with torch.no_grad(): mu0, lv0 = m.encode(feat=F)
    for p in m.parameters(): p.requires_grad_(False)
    mu = mu0.clone().requires_grad_(True); lv = lv0.clone().requires_grad_(True); opt = torch.optim.Adam([mu, lv], lr=5e-3)
    for s in range(steps):
        z = m.sample(mu, lv.clamp(-12, 4)); lam = m.decode_cells(z)
        loss = ((-(lam * St).sum(-1) + nc * m.logz(lam)).sum(-1) + m.kl(mu, lv.clamp(-12, 4))).mean(); opt.zero_grad(); loss.backward(); opt.step()
    e0 = elbo_terms(m, F, St, nc, mu0, lv0, seed=1)[0]; e1 = elbo_terms(m, F, St, nc, mu.detach(), lv.detach().clamp(-12, 4), seed=1)[0]
    for p in m.parameters(): p.requires_grad_(True)
    return (e1 - e0).mean().item(), (e1 - e0).max().item(), kl_of(m, mu.detach(), lv.detach()).mean().item()
def kl_of(m, mu, lv): return m.kl(mu, lv.clamp(-12, 4))

def decoder_frame_oracle(m, S, mu):
    """latent only fixes the frame (global director from mean-z decode); fit the multipole family directly to spins in that frame."""
    lam = m.decode_cells(mu)[:, 0]; V = frame_of(lam)                                 # [C,3,3]
    Sp = torch.einsum("cxyzi,cij->cxyzj", S.double(), V).reshape(-1, 3).numpy()
    fits = {L: Fitter(L, order=59, alpha=1e-8).fit(Sp) for L in (1, 2, 4, 6)}; f6 = fits[6]; P = f6.power()
    a = {l: f6.coef(l, 0) * np.sqrt((2 * l + 1) / (4 * np.pi)) for l in (2, 4, 6)}
    return dict(a2=a[2], a4=a[4], a6=a[6], nll={str(L): f.nll for L, f in fits.items()}, dNLL={str(L): fits[L].nll - f6.nll for L in (1, 2, 4)},
                odd_even=sum(P[l] for l in P if l % 2) / sum(P[l] for l in P if l % 2 == 0))

@torch.no_grad()
def evaluate_A(path, orc_val):
    m, a = load(path); S = torch.tensor(np.load(f"{a['data']}/ll_L12_T{a['T']:.3f}_{a['st']}.npy")); nv = max(len(S) // 10, 10); Sv = S[-nv:]
    R = a["R"]; nc = (12 // R) ** 3; F = cell_features(Sv, R); St = compute_stats(Sv, R)
    mu, lv = m.encode(feat=F); kl = m.kl(mu, lv).mean().item()
    e, nll_samp, _ = elbo_terms(m, F, St, nc, mu, lv, nsamp=16)
    lam_mean = m.decode_cells(mu); nll_mean = (-(lam_mean * St).sum(-1) + nc * m.logz(lam_mean)).sum(-1).mean().item() / 1728
    g = torch.Generator().manual_seed(5); a_samp = []; 
    for _ in range(8): a_samp.append(torch.stack(a_l(m.decode_cells(m.sample(mu, lv)))).mean((1, 2)))
    a_samp = torch.stack(a_samp).mean(0); a_mean = torch.stack(a_l(lam_mean)).mean((1, 2))
    ev, od = power_ratio(lam_mean); o = [r for r in orc_val if abs(r["T"] - a["T"]) < 1e-6 and r["st"] == a["st"]][0]
    out = dict(run=path.split("/")[-1], T=a["T"], R=R, kl=kl, nll_samp=nll_samp.mean().item() / 1728, nll_mean=nll_mean, nll_oracle_val=o["nll"]["6"],
               a2_mean=a_mean[0].item(), a2_samp=a_samp[0].item(), a4_mean=a_mean[1].item(), a4_samp=a_samp[1].item(), a6_mean=a_mean[2].item(),
               a2_oracle=o["a2"], a4_oracle=o["a4"], odd_even=(od.mean() / ev.mean()).item(),
               latent_power=[float((mu[..., s:s + w] ** 2).sum()) for s, w in [(0, 4), (4, 12), (16, 20), (36, 28)]])
    out["a2_rel_diff_mean_vs_samp"] = abs(out["a2_mean"] - out["a2_samp"]) / abs(out["a2_samp"])
    out["sigma2_mean"] = -(35 / 18) * out["a4_mean"] / out["a2_mean"] ** 2 if out["a2_mean"] > 0.3 else None
    with torch.enable_grad(): gap = amort_gap(m, F[:20], St[:20], nc)
    out["amort_gap_mean"], out["amort_gap_max"], out["kl_after_direct_opt"] = gap
    if R == 1: out["frame_oracle"] = decoder_frame_oracle(m, Sv, mu)
    out["criteria"] = dict(kl_le_30=kl <= 30, nll_within_0p005=abs(out["nll_samp"] - o["nll"]["6"]) <= 0.005, a2_mean_vs_samp_le_2pct=out["a2_rel_diff_mean_vs_samp"] <= 0.02,
                           odd_even_le_1e3=out["odd_even"] <= 1e-3, amort_gap_le_5=out["amort_gap_mean"] <= 5)
    return out
if __name__ == "__main__":
    orc = json.load(open("../results/oracle_val.json")); r = evaluate_A(sys.argv[1], orc); json.dump(r, open(sys.argv[2], "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "frame_oracle"}, indent=1))
