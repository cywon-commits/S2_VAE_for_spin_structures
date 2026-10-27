"""E5 task B2: latent-space HMC on log p(z) + log w(z;h) (joint density for hierarchical priors), field along e_z (prior is rotation invariant).
python3 run_B_mcmc.py RUN [nwarm nsamp nchain]  -> results/response/Bmc_<RUN>.json"""
import sys, json, os, time, math, numpy as np, torch
from vae_response import *
HS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
def make(m, B):
    C = m.R ** 2; mult = mult_of(m); beta = 1 / m.T; tau = m.tau_comp().detach().double() if m.kind != "1" else None
    def lam_of(zg, zc): return m.decode((zc if m.kind != "1" else zg).float())
    def logp(zg, zc):
        lp = -0.5 * (zg ** 2).sum((1, 2))
        if m.kind != "1": lp = lp - 0.5 * (((zc - zg) / tau) ** 2).sum((1, 2))
        return lp
    def logpost(zg, zc, h):
        lam = lam_of(zg, zc).double(); d = zeeman_shift(torch.tensor([0., 0., h]), beta)
        lw = mult * (torch.logsumexp((lam + d) @ YQ.T + LW, -1) - torch.logsumexp(lam @ YQ.T + LW, -1)).sum((1, 2)); return logp(zg, zc) + lw
    return logpost, lam_of
def obs(m, lam, h):
    mult = mult_of(m); beta = 1 / m.T; d = zeeman_shift(torch.tensor([0., 0., h]), beta)
    _, M = site_moments(lam.detach().double() + d); _, S = site_moments(lam.detach()); Ssum = mult * S.sum((1, 2)); w, V = torch.linalg.eigh(Ssum)
    return (mult * M[..., 2].sum((1, 2)) / N_SITES).numpy(), (V[:, 2, 0] ** 2).numpy()
def hmc(m, h, nwarm, nsamp, B, gen, nleap=15):
    logpost, lam_of = make(m, B); C = m.R ** 2
    zg = torch.randn(B, 1, 64, generator=gen, dtype=torch.float64); zc = zg + (m.tau_comp().detach().double() * torch.randn(B, C, 64, generator=gen, dtype=torch.float64) if m.kind != "1" else 0)
    eps = 1e-3; accs = []; mt = []; nt = []; ar = 0.7
    def U(a, b):
        a = a.detach().requires_grad_(True); b = b.detach().requires_grad_(True) if m.kind != "1" else a
        lp = logpost(a, b, h); g = torch.autograd.grad(lp.sum(), [a, b] if m.kind != "1" else [a])
        return lp.detach(), g
    lp, g = U(zg, zc)
    for it in range(nwarm + nsamp):
        pg = torch.randn_like(zg); pc = torch.randn_like(zc) if m.kind != "1" else None; a, b = zg.clone(), zc.clone(); ga = g[0]; gb = g[1] if m.kind != "1" else None
        H0 = lp - 0.5 * (pg ** 2).sum((1, 2)) - (0.5 * (pc ** 2).sum((1, 2)) if pc is not None else 0)
        pa = pg + 0.5 * eps * ga; pb = pc + 0.5 * eps * gb if pc is not None else None
        for l in range(nleap):
            a = a + eps * pa
            if pc is not None: b = b + eps * pb
            lp1, g1 = U(a, b if pc is not None else a)
            f = 1.0 if l < nleap - 1 else 0.5
            pa = pa + f * eps * g1[0]
            if pc is not None: pb = pb + f * eps * g1[1]
        H1 = lp1 - 0.5 * (pa ** 2).sum((1, 2)) - (0.5 * (pb ** 2).sum((1, 2)) if pc is not None else 0)
        ok = torch.isfinite(H1) & (torch.log(torch.rand(B, generator=gen, dtype=torch.float64)) < (H1 - H0)); acc = ok.double().mean().item()
        okk = ok[:, None, None]; zg = torch.where(okk, a, zg); zc = torch.where(okk, b if pc is not None else zg, zc) if pc is not None else zg; lp = torch.where(ok, lp1, lp)
        g = (torch.where(okk, g1[0], g[0]),) + ((torch.where(okk, g1[1], g[1]),) if pc is not None else ())
        if it < nwarm: eps *= math.exp(0.5 * (acc - ar))
        else:
            accs.append(acc)
            if (it - nwarm) % 2 == 0:
                mm, nn_ = obs(m, lam_of(zg, zc), h); mt.append(mm); nt.append(nn_)
    mt = np.array(mt); nt = np.array(nt)
    x = mt.mean(1) - mt.mean(); n = len(x); ac = [np.sum(x[:n - k] * x[k:]) / np.sum(x * x) for k in range(1, min(100, n // 2))]; tau_int = 1 + 2 * np.sum(np.array(ac)[: next((i for i, v in enumerate(ac) if v < 0.05), len(ac))])
    # per-chain spread of the means as a convergence diagnostic
    return dict(h=h, m=float(mt[len(mt) // 2:].mean()), m_chain_sd=float(mt[len(mt) // 2:].mean(0).std()), nuh2=float(nt[len(nt) // 2:].mean()), acc=float(np.mean(accs)), eps=float(eps), tau_int=float(tau_int), n_obs=len(mt))
if __name__ == "__main__":
    run = sys.argv[1]; nwarm = int(sys.argv[2]) if len(sys.argv) > 2 else 300; nsamp = int(sys.argv[3]) if len(sys.argv) > 3 else 400; B = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    m = load_kag(f"{ROOT}/results/kag/{run}"); out = []; gen = torch.Generator().manual_seed(3)
    for h in (HS if len(sys.argv) <= 5 else [float(x) for x in sys.argv[5].split(",")]):
        t0 = time.time(); r = hmc(m, h, nwarm, nsamp, B, gen); r["run"] = run; out.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, f"{time.time()-t0:.0f}s", flush=True)
        json.dump(out, open(f"{ROOT}/results/response/Bmc_{run[:-3]}.json", "w"), indent=1)
