"""E3 v2 A: site-resolved decoder.  The decoder receives the cell latent z_c and the position of the site inside the cell (sublattice s, unit-cell coordinates (bx,by));
the position enters only as a learned rotation-invariant scalar embedding (l=0 channels) in the tensor products, so SO(3) equivariance is kept.
Likelihood per site: sum_i [-lam_i . Y(n_i) + log Z(lam_i)].  Encoder and up/down prior as KVAE5."""
import torch, torch.nn as nn
from e3nn import o3
from kmodel import KVAE, LAT, sh, LMAX
from kmodel5 import KVAE5
OUT1 = o3.Irreps("+".join(f"1x{l}e" for l in range(1, LMAX + 1)))        # one site, 80 dims

def site_Y(S, R, sign=1.0):
    """S [B,L,L,3,3] -> Y [B,R^2,P,80] with P = b*b*3 positions ordered (bx,by,s); time reversal: Y_l(-n) = (-1)^l Y_l(n)."""
    B, L = S.shape[:2]; b = L // R; Y = sh(range(1, LMAX + 1), S.double()).float()
    if sign < 0: Y = Y * torch.cat([torch.full((2 * l + 1,), (-1.0) ** l) for l in range(1, LMAX + 1)])
    return Y.reshape(B, R, b, R, b, 3, -1).permute(0, 1, 3, 2, 4, 5, 6).reshape(B, R * R, b * b * 3, -1)

class KVAE6(KVAE5):
    def __init__(self, R=4, K=12, dec_hid=4, **kw):
        super().__init__(R=R, dec_hid=dec_hid, **kw)
        b = 12 // R; self.P = b * b * 3; self.K = K; self.emb = nn.Parameter(torch.randn(self.P, K) / K ** 0.5)
        outk = o3.Irreps("+".join(f"{K}x{l}e" for l in range(1, LMAX + 1))); self.outk = outk
        for n in ("d2", "dlin"): delattr(self, n)
        hid = self.d1.irreps_out; self.e2 = o3.FullyConnectedTensorProduct(hid, LAT, outk); self.elin = o3.Linear(LAT, outk)
    def decode(self, z):                                   # z [B,C,dim] -> lam [B,C,P,80]; K cell-level multipole sets mixed by the invariant position embedding
        z = z * self.zmask; o = 0.01 * (self.e2(self.d1(z, z), z) + self.elin(z)); parts = []; off = 0
        for l in range(1, LMAX + 1):
            w = self.K * (2 * l + 1); parts.append(o[..., off:off + w].reshape(*o.shape[:-1], self.K, 2 * l + 1)); off += w
        o = torch.cat(parts, -1)                           # [B,C,K,80]
        return torch.einsum("pk,bcky->bcpy", self.emb, o)
    def nll_cfg(self, lam, Y, nsub=None):                  # Y [B,C,P,80]
        return (-(lam * Y).sum(-1) + self.logz(lam)).sum((-1, -2))
    def forward_stats(self, feat, Y, nsub=None, beta=1.0, alpha=0.0, sample=True):
        mu_g, lv_g = self.encode(feat.mean(1, keepdim=True)); mu_c, lv_c = self.encode(feat)
        z = self.sample(mu_c, lv_c) if sample else mu_c; lam = self.decode(z); nll = self.nll_cfg(lam, Y); N = 3 * self.L ** 2
        kl_g, kl_c = self.kl_terms(mu_g, lv_g, mu_c, lv_c); kl = kl_g + kl_c; reg = alpha * (lam ** 2 * self.pen).sum(-1).mean()
        return dict(loss=nll.mean() / N + beta * kl.mean() / N + reg, nll=nll.mean() / N, kl=kl.mean(), kl_g=kl_g.mean(), kl_c=kl_c.mean(), lam=lam, mu=mu_c, lv=lv_c, mu_g=mu_g, lv_g=lv_g)

def _kl_terms_r1(self, mu_g, lv_g, mu_c, lv_c):
    if self.R == 1:                                        # single global latent: plain N(0,I) prior, no cell level
        kg = self.kl(mu_g, lv_g); return kg, torch.zeros_like(kg)
    return KVAE5.kl_terms(self, mu_g, lv_g, mu_c, lv_c)
KVAE6.kl_terms = _kl_terms_r1
