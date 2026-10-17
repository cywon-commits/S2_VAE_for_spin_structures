"""E3 B: bottom-up posterior, top-down prior for the kagome VAE.  q(z_g|x) and q(z_c|x_c) come from the same encoder head (fully pooled / cell-pooled features);
p(z_g)=N(0,I), p(z_c|z_g)=N(z_g, tau_l^2) with tau_l learned per order; closed-form KL (expectation over q(z_g) is analytic)."""
import torch, torch.nn as nn
from kmodel import KVAE

class KVAE5(KVAE):
    def __init__(self, R=3, tau0=0.5, **kw):
        super().__init__(R=R, **kw)
        self.log_tau = nn.Parameter(torch.full((4,), float(torch.log(torch.tensor(tau0)))))
        ls = []
        for l in range(4): ls += [l] * (4 * (2 * l + 1))
        self.register_buffer("lidx", torch.tensor(ls))
    def tau_comp(self): return self.log_tau.exp()[self.lidx]
    def kl_terms(self, mu_g, lv_g, mu_c, lv_c):
        kl_g = self.kl(mu_g, lv_g); lvg = self.expand_logvar(lv_g); lvc = self.expand_logvar(lv_c); tau = self.tau_comp()
        kl_c = ((torch.log(tau) - 0.5 * lvc + (lvc.exp() + (mu_c - mu_g) ** 2 + lvg.exp()) / (2 * tau ** 2) - 0.5) * self.zmask).sum((-1, -2))
        return kl_g, kl_c
    def forward_stats(self, feat, stats, nsub, beta=1.0, alpha=0.0, sample=True):
        mu_g, lv_g = self.encode(feat.mean(1, keepdim=True)); mu_c, lv_c = self.encode(feat)
        z = self.sample(mu_c, lv_c) if sample else mu_c; lam = self.decode(z); nll = self.nll_cfg(lam, stats, nsub); N = 3 * self.L ** 2
        kl_g, kl_c = self.kl_terms(mu_g, lv_g, mu_c, lv_c); kl = kl_g + kl_c; reg = alpha * (lam ** 2 * self.pen).sum(-1).mean()
        return dict(loss=nll.mean() / N + beta * kl.mean() / N + reg, nll=nll.mean() / N, kl=kl.mean(), kl_g=kl_g.mean(), kl_c=kl_c.mean(), lam=lam, mu=mu_c, lv=lv_c, mu_g=mu_g, lv_g=lv_g)
