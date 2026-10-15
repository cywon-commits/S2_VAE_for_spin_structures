"""v4 task A: bottom-up posterior, top-down prior.
 q(z_g|x) = N(mu_g, s_g^2)        (same encoder head applied to the fully pooled features)
 q(z_c|x_c) = N(mu_c, s_c^2)      (same encoder head applied to the cell-pooled features; ABSOLUTE cell latent)
 p(z_g) = N(0, I),  p(z_c | z_g) = N(z_g, tau_l^2 I)   (tau_l learned per order l)
 ELBO = E_q log p(x|z_c) - KL(q(z_g)||p(z_g)) - sum_c E_{q(z_g)} KL(q(z_c)||p(z_c|z_g));  the z_g expectation is closed form."""
import torch, torch.nn as nn
from mvae2 import MVAE2

class MVAE5(MVAE2):
    def __init__(self, L=12, R=2, tau0=0.5, **kw):
        super().__init__(L=L, R=R, **kw)
        self.log_tau = nn.Parameter(torch.full((4,), float(torch.log(torch.tensor(tau0)))))
        ls = []
        for l in range(4): ls += [l] * (4 * (2 * l + 1))
        self.register_buffer("lidx", torch.tensor(ls))
    def tau_comp(self): return self.log_tau.exp()[self.lidx]
    def encode_g(self, feat): return super().encode(feat=feat.mean(1, keepdim=True))
    def encode_c(self, feat): return super().encode(feat=feat)
    def kl_terms(self, mu_g, lv_g, mu_c, lv_c):
        kl_g = self.kl(mu_g, lv_g)
        lvg = self.expand_logvar(lv_g); lvc = self.expand_logvar(lv_c); tau = self.tau_comp()
        kl_c = ((torch.log(tau) - 0.5 * lvc + (lvc.exp() + (mu_c - mu_g) ** 2 + lvg.exp()) / (2 * tau ** 2) - 0.5) * self.zmask).sum((-1, -2))
        return kl_g, kl_c
    def forward_stats(self, feat, stats, ncell, beta=1.0, alpha=0.0, sample=True):
        mu_g, lv_g = self.encode_g(feat); mu_c, lv_c = self.encode_c(feat)
        z = self.sample(mu_c, lv_c) if sample else mu_c
        lam = self.decode_cells(z)
        nll_cfg = (-(lam * stats).sum(-1) + ncell * self.logz(lam)).sum(-1); N = self.L ** 3
        pen = torch.tensor([l * (l + 1) for l in range(1, 7) for _ in range(2 * l + 1)], dtype=lam.dtype)
        reg = alpha * (lam ** 2 * pen).sum(-1).mean()
        kl_g, kl_c = self.kl_terms(mu_g, lv_g, mu_c, lv_c); kl = kl_g + kl_c
        return dict(loss=nll_cfg.mean() / N + beta * kl.mean() / N + reg, nll=nll_cfg.mean() / N, kl=kl.mean(), kl_g=kl_g.mean(), kl_c=kl_c.mean(),
                    lam=lam, mu=mu_c, lv=lv_c, mu_g=mu_g, lv_g=lv_g)
