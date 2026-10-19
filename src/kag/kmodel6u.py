"""E3 v4 A: site-resolved decoder with a cap-direction input u_{c_i} (unit vector, l=1 input) per site.  Cap term enters the site output through (a) a tensor product of Y_{1..3}(u_i) with the cell latent
(equivariant, can displace/rotate the cap centre by the latent's l=1.. content) and (b) a rotation-covariant cap lam_l = s_l * Y_l(u_i) with learned shape scalars s_l (position + l=0 latent dependent);
the colour-free low-rank pathway of kmodel6 is kept.  u is a physical vector (transforms like the spins), so equivariance is exact."""
import torch, torch.nn as nn
from e3nn import o3
from kmodel import LMAX, LAT, sh
from kmodel6 import KVAE6, OUT1
UIN = o3.Irreps("1x1e+1x2e+1x3e")
def u_feat(u):                                       # u [...,3] -> Y_1..3(u) 'component' normalisation [...,15]
    return sh([1, 2, 3], u, "component")
def u_cap(u):                                        # Y_l(u), l=1..8, same (integral) normalisation as the likelihood Y [...,80]
    return sh(range(1, LMAX + 1), u)
class KVAE6U(KVAE6):
    def __init__(self, R=4, K=12, **kw):
        super().__init__(R=R, K=K, **kw)
        ZR = o3.Irreps("2x0e+2x1e+2x2e+2x3e"); self.zproj = o3.Linear(LAT, ZR); self.tp_u = o3.FullyConnectedTensorProduct(UIN, ZR, OUT1)
        self.s0 = nn.Parameter(torch.zeros(self.P, LMAX)); self.s1 = nn.Parameter(torch.zeros(4, LMAX))
        with torch.no_grad(): self.s0[:, 0] = 1.26                    # initial cap ~ vMF(kappa=30) around u:  lam_1m = kappa * 4pi/3 (x100 scale below)
        lidx = torch.cat([torch.full((2 * l + 1,), l - 1) for l in range(1, LMAX + 1)]); self.register_buffer("cap_l", lidx)
    def decode(self, z, u):                                # z [B,C,dim], u [B,C,P,3]
        zm = z * self.zmask; base = KVAE6.decode(self, z)  # [B,C,P,80]
        B, C, P, _ = u.shape; zr = self.zproj(zm); zz = zr.unsqueeze(2).expand(B, C, P, zr.shape[-1]).reshape(B * C * P, -1)
        t = 0.1 * self.tp_u(u_feat(u).reshape(B * C * P, -1), zz).reshape(B, C, P, -1)
        s = 100.0 * (self.s0.unsqueeze(0).unsqueeze(0) + (zm[..., :4] @ self.s1).unsqueeze(2))          # [B,C,P,8]
        return base + t + s[..., self.cap_l] * u_cap(u)
    def forward_stats(self, feat, Y, u, nsub=None, beta=1.0, alpha=0.0, sample=True):
        mu_g, lv_g = self.encode(feat.mean(1, keepdim=True)); mu_c, lv_c = self.encode(feat)
        z = self.sample(mu_c, lv_c) if sample else mu_c; lam = self.decode(z, u); nll = self.nll_cfg(lam, Y); N = 3 * self.L ** 2
        kl_g, kl_c = self.kl_terms(mu_g, lv_g, mu_c, lv_c); kl = kl_g + kl_c; reg = alpha * (lam ** 2 * self.pen).sum(-1).mean()
        return dict(loss=nll.mean() / N + beta * kl.mean() / N + reg, nll=nll.mean() / N, kl=kl.mean(), kl_g=kl_g.mean(), kl_c=kl_c.mean(), lam=lam, mu=mu_c, lv=lv_c, mu_g=mu_g, lv_g=lv_g)
