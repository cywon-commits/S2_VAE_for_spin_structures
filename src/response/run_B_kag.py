"""E5 task B1 (kagome): exact importance-reweighting of prior samples to the Zeeman-field ensemble p_H(z) ∝ p(z) prod_i Z(lam_i + beta H.n)/Z(lam_i).
python3 run_B_kag.py RUN [K]  -> results/response/B_<RUN>.npz  (logw [K,H], mproj [K,H], nuh2 [K,H], ...)"""
import sys, os, time, numpy as np, torch
from vae_response import *
HGRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 0.002, 0.004, 0.006, 0.008, 0.011, 0.015, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
@torch.no_grad()
def lz_mean(lam, chunk=4000):
    """lam [n,80] -> logZ [n], mean [n,3]"""
    n = len(lam); LZ = torch.empty(n, dtype=torch.float64); M = torch.empty(n, 3, dtype=torch.float64)
    for i in range(0, n, chunk):
        a = lam[i:i + chunk].double() @ YQ.T + LW; LZ[i:i + chunk] = torch.logsumexp(a, -1); M[i:i + chunk] = torch.softmax(a, -1) @ XQ
    return LZ, M
if __name__ == "__main__":
    run = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 2000; m = load_kag(f"{ROOT}/results/kag/{run}"); mult = mult_of(m); beta = 1 / m.T
    gen = torch.Generator().manual_seed(1); hh = torch.tensor(HGRID, dtype=torch.float64); H = len(hh)
    z = sample_prior(m, K, gen); hat = torch.randn(K, 3, generator=gen, dtype=torch.float64); hat /= hat.norm(dim=-1, keepdim=True)         # lab-fixed field, random global rotation per sample
    u = prior_u(m, K, gen) if m.kind == "6u" else None
    logw = torch.zeros(K, H, dtype=torch.float64); mpj = torch.zeros(K, H, dtype=torch.float64); nuh2 = torch.zeros(K, dtype=torch.float64); m0 = torch.zeros(K, dtype=torch.float64); t0 = time.time(); ch = 100
    for i in range(0, K, ch):
        lam = decode(m, z[i:i + ch], u[i:i + ch] if u is not None else None); k, C, P, _ = lam.shape; flat = lam.reshape(-1, 80)
        LZ0, M0 = lz_mean(flat); S = site_moments(lam)[1]; Ssum = mult * S.sum((1, 2)); w_, V = torch.linalg.eigh(Ssum); nu = V[:, :, 0]; nuh2[i:i + k] = (nu * hat[i:i + k]).sum(-1) ** 2
        m0[i:i + k] = mult * (M0.reshape(k, C * P, 3) * hat[i:i + k, None, :]).sum((1, 2)) / N_SITES
        for j, h in enumerate(HGRID):
            d = zeeman_shift(h * hat[i:i + k], beta)                                                                              # [k,80]
            lam_h = (lam.double() + d[:, None, None, :]).reshape(-1, 80); LZ, M = lz_mean(lam_h)
            logw[i:i + k, j] = mult * (LZ - LZ0).reshape(k, C * P).sum(1)
            mpj[i:i + k, j] = mult * (M.reshape(k, C * P, 3) * hat[i:i + k, None, :]).sum((1, 2)) / N_SITES
        print(run, i + k, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(f"{ROOT}/results/response/B_{run[:-3]}.npz", h=np.array(HGRID), logw=logw.numpy(), mproj=mpj.numpy(), nuh2=nuh2.numpy(), m0=m0.numpy(), hat=hat.numpy(), T=m.T, R=m.R)
