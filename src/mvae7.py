"""v4 extra: up-posterior / down-prior with equivariant SUBSPACE covariance on the main l=2 channel, for q(z_g), q(z_c) and the prior p(z_c|z_g)
(rotation / amplitude / biaxial subspaces defined by the posterior-mean 5-vector; prior subspaces from mu_g).  Other channels: isotropic as in MVAE5.
KL(q(z_c)||p(z_c|z_g)) with E over q(z_g) is closed form for the Gaussian block:  1/2[ tr(Sp^-1 (Sq+Sg)) + (mu_c-mu_g)^T Sp^-1 (mu_c-mu_g) - 5 + ln|Sp| - ln|Sq| ]."""
import torch, torch.nn as nn
from mvae4 import MVAE4

class MVAE7(MVAE4):
    def __init__(self, L=12, R=2, tau0=0.5, **kw):
        super().__init__(L=L, R=R, **kw)
        self.log_tau = nn.Parameter(torch.full((4,), float(torch.log(torch.tensor(tau0)))))                 # isotropic tau_l (other channels)
        self.log_tau3 = nn.Parameter(torch.full((3,), float(torch.log(torch.tensor(tau0)))))                # tau_rot, tau_amp, tau_bi (main channel)
        ls = []
        for l in range(4): ls += [l] * (4 * (2 * l + 1))
        self.register_buffer("lidx", torch.tensor(ls))
    def tau_comp(self): return self.log_tau.exp()[self.lidx]
    def forward_stats(self, feat, stats, ncell, beta=1.0, alpha=0.0, sample=True):
        mu_g, lv_g, la_g = self.encode3(feat.mean(1, keepdim=True)); mu_c, lv_c, la_c = self.encode3(feat)
        z = self.sample3(mu_c, lv_c, la_c) if sample else mu_c * self.zmask
        lam = self.decode_cells(z); nll_cfg = (-(lam * stats).sum(-1) + ncell * self.logz(lam)).sum(-1); N = self.L ** 3
        pen = torch.tensor([l * (l + 1) for l in range(1, 7) for _ in range(2 * l + 1)], dtype=lam.dtype); reg = alpha * (lam ** 2 * pen).sum(-1).mean()
        kl_g = self.kl3(mu_g, lv_g, la_g); kl_c = self.kl_cell(mu_g, lv_g, la_g, mu_c, lv_c, la_c); kl = kl_g + kl_c
        return dict(loss=nll_cfg.mean() / N + beta * kl.mean() / N + reg, nll=nll_cfg.mean() / N, kl=kl.mean(), kl_g=kl_g.mean(), kl_c=kl_c.mean(), lam=lam, mu=mu_c)
    def kl_cell(self, mu_g, lv_g, la_g, mu_c, lv_c, la_c):
        a = self.act; tau = self.tau_comp(); zm = self.zmask.clone(); zm[a:a + 5] = 0
        lvg = self.expand_logvar(lv_g); lvc = self.expand_logvar(lv_c)
        kl_other = ((torch.log(tau) - 0.5 * lvc + (lvc.exp() + (mu_c - mu_g) ** 2 + lvg.exp()) / (2 * tau ** 2) - 0.5) * zm).sum((-1, -2))
        Pg = self.projectors(mu_g[..., a:a + 5]); Pc = self.projectors(mu_c[..., a:a + 5])                  # tuples (rot, amp, bi) of [...,5,5]
        t = self.log_tau3.exp(); dims = torch.tensor([2., 1., 2.])
        Spinv = sum(P / t[i] ** 2 for i, P in enumerate(Pg))                                                  # [B,1,5,5]
        Sq = sum(P * la_c.exp()[..., i:i + 1, None] for i, P in enumerate(Pc))                                # [B,C,5,5]
        Sg = sum(P * la_g.exp()[..., i:i + 1, None] for i, P in enumerate(Pg))                                # [B,1,5,5]
        v = (mu_c - mu_g)[..., a:a + 5]
        tr = ((Spinv * (Sq + Sg)).sum((-1, -2)))
        maha = torch.einsum("...i,...ij,...j->...", v, Spinv.expand(*v.shape[:-1], 5, 5), v)
        logdet = (dims * torch.log(t ** 2)).sum() - (dims * la_c).sum(-1)
        return kl_other + (0.5 * (tr + maha - 5 + logdet)).sum(-1)
