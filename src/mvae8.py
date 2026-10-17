"""v5 1.3 ablation: cell deviations restricted to the rotation + amplitude subspaces of the main l=2 channel; the biaxial part of z_c is tied to the global latent
(z_c,bi = z_g,bi, i.e. p(delta_bi)=delta(0)).  Subspaces/projectors come from mu_g (shared by posterior and prior).  Other channels: as MVAE5.
KL_main = 1/2 sum_{b in rot,amp} [ d_b ( tau_b^-2 (s_b^2 + sg_b^2) - 1 + ln tau_b^2 - ln s_b^2 ) + tau_b^-2 |P_b (m_c - mu_g)|^2 ]   (E over q(z_g) closed form)."""
import torch
from mvae7 import MVAE7

class MVAE8(MVAE7):
    def forward_stats(self, feat, stats, ncell, beta=1.0, alpha=0.0, sample=True):
        mu_g, lv_g, la_g = self.encode3(feat.mean(1, keepdim=True)); mu_c, lv_c, la_c = self.encode3(feat); a = self.act
        Prot, Pamp, Pbi = self.projectors(mu_g[..., a:a + 5])                                           # [B,1,5,5]
        zg = self.sample3(mu_g, lv_g, la_g) if sample else mu_g * self.zmask
        m5 = mu_c[..., a:a + 5]; sd = torch.exp(0.5 * la_c)
        eps = torch.randn_like(m5) if sample else torch.zeros_like(m5)
        zm = torch.einsum("...ij,...j->...i", Prot + Pamp, m5) + sd[..., 0:1] * torch.einsum("...ij,...j->...i", Prot, eps) + sd[..., 1:2] * torch.einsum("...ij,...j->...i", Pamp, eps) \
            + torch.einsum("...ij,...j->...i", Pbi, zg[..., a:a + 5])
        z = (self.sample(mu_c, lv_c) if sample else mu_c * self.zmask).clone(); z[..., a:a + 5] = zm
        lam = self.decode_cells(z); nll_cfg = (-(lam * stats).sum(-1) + ncell * self.logz(lam)).sum(-1); N = self.L ** 3
        pen = torch.tensor([l * (l + 1) for l in range(1, 7) for _ in range(2 * l + 1)], dtype=lam.dtype); reg = alpha * (lam ** 2 * pen).sum(-1).mean()
        kl_g = self.kl3(mu_g, lv_g, la_g); kl_c = self.kl_cell_abl(mu_g, lv_g, la_g, mu_c, lv_c, la_c, (Prot, Pamp)); kl = kl_g + kl_c
        return dict(loss=nll_cfg.mean() / N + beta * kl.mean() / N + reg, nll=nll_cfg.mean() / N, kl=kl.mean(), kl_g=kl_g.mean(), kl_c=kl_c.mean(), lam=lam, mu=mu_c)
    def kl_cell_abl(self, mu_g, lv_g, la_g, mu_c, lv_c, la_c, P):
        a = self.act; tau = self.tau_comp(); zm = self.zmask.clone(); zm[a:a + 5] = 0
        lvg = self.expand_logvar(lv_g); lvc = self.expand_logvar(lv_c)
        kl_other = ((torch.log(tau) - 0.5 * lvc + (lvc.exp() + (mu_c - mu_g) ** 2 + lvg.exp()) / (2 * tau ** 2) - 0.5) * zm).sum((-1, -2))
        t = self.log_tau3.exp(); dims = (2., 1.); v = (mu_c - mu_g)[..., a:a + 5]; out = 0.
        for b in range(2):
            pv = torch.einsum("...ij,...j->...i", P[b].expand(*v.shape[:-1], 5, 5), v)
            s2 = la_c[..., b].exp(); sg2 = la_g[..., b].exp()
            out = out + 0.5 * (dims[b] * ((s2 + sg2) / t[b] ** 2 - 1 + torch.log(t[b] ** 2) - torch.log(s2)) + (pv ** 2).sum(-1) / t[b] ** 2)
        return kl_other + out.sum(-1)
