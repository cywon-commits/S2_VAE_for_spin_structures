"""E7 task 1: diagnostics (1-a) of aggregated posterior vs prior, rotation-augmented mixture refit (1-b)."""
import sys, os, json, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nem_response import *
from e6_response import projectors
ACT = {"A": [21, 22, 23, 24, 25, 31, 32, 33, 34, 35]}
@torch.no_grad()
def posterior_parts(m, Sv, gen):
    F = cell_features(Sv, m.R)
    if m.cls == "MVAE3":
        mug, lvg = m.encode_g(F); md, lvd = m.encode_d(F); zg = m.sample(mug, lvg); dl = m.sample(md, lvd); zc = zg + dl
    else:
        mu_g, lv_g, la_g = m.encode3(F.mean(1, keepdim=True)); mu_c, lv_c, la_c = m.encode3(F); a = m.act
        Prot, Pamp, Pbi = m.projectors(mu_g[..., a:a + 5]); m5 = mu_c[..., a:a + 5]; sd = torch.exp(0.5 * la_c); eps = torch.randn_like(m5); zg = m.sample3(mu_g, lv_g, la_g)
        zm = torch.einsum("...ij,...j->...i", Prot + Pamp, m5) + sd[..., 0:1] * torch.einsum("...ij,...j->...i", Prot, eps) + sd[..., 1:2] * torch.einsum("...ij,...j->...i", Pamp, eps) + torch.einsum("...ij,...j->...i", Pbi, zg[..., a:a + 5])
        zc = m.sample(mu_c, lv_c).clone(); zc[..., a:a + 5] = zm
    return zg * m.zmask, zc * m.zmask
@torch.no_grad()
def eps_of(m, zg, zc):
    """non-centred cell noise: (zc-zg)/tau (all channels; MVAE8 main channel: rot/amp parts separately scaled)."""
    tau = m.tau_comp().detach().clone(); e = (zc - zg) / tau.clamp_min(1e-9) * m.zmask
    if m.cls == "MVAE8":
        a = m.act; tr, ta, _ = m.log_tau3.exp().detach(); Pr, Pa, _ = m.projectors(zg[..., a:a + 5]); d5 = (zc - zg)[..., a:a + 5]
        e[..., a:a + 5] = torch.einsum("kij,kcj->kci", Pr[:, 0], d5) / tr + torch.einsum("kij,kcj->kci", Pa[:, 0], d5) / ta
    return e
@torch.no_grad()
def S_of(m, z, chunk=500):
    mult = (12 // m.R) ** 3; out = []
    for i in range(0, len(z), chunk):
        p = probs(m.decode_cells(z[i:i + chunk])); S2 = torch.einsum("kcq,qa,qb->kab", p, XQ, XQ) * mult; w, V = torch.linalg.eigh(S2); d = V[:, :, 2]; f = 1.5 * (d @ XQ.T) ** 2 - 0.5
        out.append(mult * torch.einsum("kcq,kq->k", p, f) / N)
    return torch.cat(out)
def data_S(T):
    S = val_spins(T).numpy().reshape(-1, N, 3).astype(float); out = []
    for X in S:
        Q = 1.5 * X.T @ X / N - 0.5 * np.eye(3); w, V = np.linalg.eigh(Q); out.append(np.mean(1.5 * (X @ V[:, 2]) ** 2 - 0.5))
    return np.array(out)
def cond_cells(m, zg, gen):
    """cells from the conditional prior given z_g"""
    C = m.R ** 3; B = len(zg); e = torch.randn(B, C, 64, generator=gen)
    from run_E6_B import to_z
    return to_z(m, zg, e)

# ------------------------------------------------------------------ 1-b: rotation-augmented mixture prior on z_g
from e3nn import o3
ACTIVE = [21, 22, 23, 24, 25, 31, 32, 33, 34, 35]
def rotate_l2(zg_act, gen, K_rot=16):
    """zg_act [n,10] (two l=2 channels, e3nn real basis) -> [n*K_rot,10], same random SO(3) rotation on both channels"""
    n = len(zg_act); out = []
    for _ in range(K_rot):
        a = torch.rand(n, generator=gen) * 2 * math.pi; b = torch.acos(1 - 2 * torch.rand(n, generator=gen)); c = torch.rand(n, generator=gen) * 2 * math.pi
        D = o3.wigner_D(2, a, b, c).to(zg_act.dtype)                                                  # [n,5,5]
        out.append(torch.cat([torch.einsum("nij,nj->ni", D, zg_act[:, :5]), torch.einsum("nij,nj->ni", D, zg_act[:, 5:])], 1))
    return torch.cat(out)
def fit_gmm(x, ks=(20, 30, 40, 50), seed=0, reg=1e-3):
    from sklearn.mixture import GaussianMixture
    best = None; bics = {}
    for k in ks:
        g = GaussianMixture(k, covariance_type="full", reg_covar=reg, max_iter=300, random_state=seed).fit(x); b = g.bic(x); bics[k] = float(b)
        if best is None or b < best[0]: best = (b, g, k)
    return best[1], best[2], bics
class GMMPrior:
    def __init__(self, g):
        self.w = torch.tensor(g.weights_, dtype=torch.float64).log(); self.mu = torch.tensor(g.means_, dtype=torch.float64); self.L = torch.tensor(g.precisions_cholesky_, dtype=torch.float64); self.g = g
        self.ld = torch.log(torch.diagonal(self.L, dim1=1, dim2=2)).sum(1)
    def logpdf(self, x):
        """x [B,10] -> log density [B] (differentiable)"""
        d = x[:, None, :] - self.mu[None]; y = torch.einsum("kbi,kij->kbj", d.transpose(0, 1), self.L); return torch.logsumexp(self.w[:, None] + self.ld[:, None] - 0.5 * (y ** 2).sum(-1) - 0.5 * x.shape[1] * math.log(2 * math.pi), 0)
    def sample(self, n, seed):
        self.g.random_state = seed; return torch.tensor(self.g.sample(n)[0], dtype=torch.float32)
def sample_prior_new(m, K, gen, gp, seed=0):
    """z_g from the mixture (active comps), other comps zero; cells from the conditional prior (non-centred map of run_E6_B.to_z)"""
    from run_E6_B import to_z
    zg = torch.zeros(K, 1, 64); zg[:, 0, ACTIVE] = gp.sample(K, seed); return to_z(m, zg, torch.randn(K, m.R ** 3, 64, generator=gen)), zg

# ------------------------------------------------------------------ 1-b': cell-level rescaling of the non-centred noise (posterior-matched variances)
def eps_scales(m, zg, zc):
    """per-component std of the non-centred cell noise of the aggregated posterior (non-main channels) and (std_rot, std_amp) for MVAE8's main channel"""
    e = eps_of(m, zg, zc); fac = e.reshape(-1, 64).var(0).sqrt().clamp_min(1e-3) * m.zmask + (1 - m.zmask)
    fr = fa = 1.0
    if m.cls == "MVAE8":
        a = m.act; Pr, Pa, _ = m.projectors(zg[..., a:a + 5]); e5 = e[..., a:a + 5]; er = torch.einsum("kij,kcj->kci", Pr[:, 0], e5); ea = torch.einsum("kij,kcj->kci", Pa[:, 0], e5)
        fr = float((er ** 2).sum(-1).mean() / 2) ** 0.5; fa = float((ea ** 2).sum(-1).mean()) ** 0.5
    return fac, fr, fa
def to_z_scaled(m, zg, eps, sc):
    fac, fr, fa = sc; tau = m.tau_comp().detach(); z = zg + tau * fac * eps
    if m.cls == "MVAE8":
        a = m.act; tr, ta, _ = m.log_tau3.exp().detach(); Pr, Pa, _ = m.projectors(zg[..., a:a + 5].detach()); e5 = eps[..., a:a + 5]
        z = z.clone(); z[..., a:a + 5] = zg[..., a:a + 5] + tr * fr * torch.einsum("kij,kcj->kci", Pr[:, 0], e5) + ta * fa * torch.einsum("kij,kcj->kci", Pa[:, 0], e5)
    return z * m.zmask
