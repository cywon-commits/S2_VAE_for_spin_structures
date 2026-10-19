"""E3 v3 A: site-resolved decoder (kmodel6) with an extra conditioning input: the oracle colour of each site (one-hot, 3, rotation-invariant scalar).  Encoder / latent unchanged,
colours are not part of the latent and not in the KL.  The colour enters the per-site mixing weights of the K cell-level multipole sets: w_{p,k} = emb[p,k] + onehot(c_p) . Wc[:,k]."""
import numpy as np, torch, torch.nn as nn
from kmodel6 import KVAE6, site_Y
from ring_oracle import global_triad

def oracle_colors(S):
    """S [C,12,12,3,3] -> colour index (0,1,2) per site via the global triad of each configuration (kagome/loops.py rule), int64 [C,12,12,3]."""
    out = np.zeros(S.shape[:-1], np.int64)
    for i in range(len(S)):
        X = S[i].reshape(-1, 3).astype(float); U = global_triad(X); out[i] = np.argmax(X @ U.T, 1).reshape(S.shape[1:-1])
    return out
def site_col(col, R):
    """col [B,L,L,3] int -> one-hot [B,R^2,P,3] with the site ordering of site_Y."""
    B, L = col.shape[:2]; b = L // R; c = torch.nn.functional.one_hot(torch.as_tensor(col), 3).float()
    return c.reshape(B, R, b, R, b, 3, 3).permute(0, 1, 3, 2, 4, 5, 6).reshape(B, R * R, b * b * 3, 3)

class KVAE6C(KVAE6):
    def __init__(self, R=4, K=12, **kw):
        super().__init__(R=R, K=K, **kw); self.Wc = nn.Parameter(0.05 * torch.randn(3, K))
    def decode(self, z, col):                              # z [B,C,dim], col [B,C,P,3]
        z = z * self.zmask; o = 0.01 * (self.e2(self.d1(z, z), z) + self.elin(z)); parts = []; off = 0
        from kmodel import LMAX
        for l in range(1, LMAX + 1):
            w = self.K * (2 * l + 1); parts.append(o[..., off:off + w].reshape(*o.shape[:-1], self.K, 2 * l + 1)); off += w
        o = torch.cat(parts, -1)                           # [B,C,K,80]
        w = self.emb + col @ self.Wc                       # [B,C,P,K]
        return torch.einsum("bcpk,bcky->bcpy", w, o)
    def forward_stats(self, feat, Y, col, nsub=None, beta=1.0, alpha=0.0, sample=True):
        mu_g, lv_g = self.encode(feat.mean(1, keepdim=True)); mu_c, lv_c = self.encode(feat)
        z = self.sample(mu_c, lv_c) if sample else mu_c; lam = self.decode(z, col); nll = self.nll_cfg(lam, Y); N = 3 * self.L ** 2
        kl_g, kl_c = self.kl_terms(mu_g, lv_g, mu_c, lv_c); kl = kl_g + kl_c; reg = alpha * (lam ** 2 * self.pen).sum(-1).mean()
        return dict(loss=nll.mean() / N + beta * kl.mean() / N + reg, nll=nll.mean() / N, kl=kl.mean(), kl_g=kl_g.mean(), kl_c=kl_c.mean(), lam=lam, mu=mu_c, lv=lv_c, mu_g=mu_g, lv_g=lv_g)
