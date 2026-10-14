"""v3 2.2: hierarchical prior.  z_c = z_g + delta_c ;  z_g ~ N(0,I),  delta_c ~ N(0, tau_l^2 I)  (tau_l learned per order l, equivariant).
Posterior q(z_g) q(delta_c): q(z_g) from the R=1 (pooled) path of the v2 encoder; q(delta_c) from a second cell network on the cell features."""
import torch, torch.nn as nn
from e3nn import o3
from mvae import LAT_IRREPS
from mvae2 import MVAE2, FCLayer, FEAT_IRREPS

class MVAE3(MVAE2):
    def __init__(self, L=12, R=2, cell_ch=8, dec_hid=4, order=59, tau0=0.3):
        super().__init__(L=L, R=R, cell_ch=cell_ch, dec_hid=dec_hid, order=order)
        self.d1c = FCLayer(FEAT_IRREPS, cell_ch); self.d2c = FCLayer(self.d1c.gate.irreps_out, cell_ch); h = self.d2c.gate.irreps_out
        self.mud = o3.Linear(h, LAT_IRREPS); self.lvd = o3.FullyConnectedTensorProduct(h, h, o3.Irreps(f"{LAT_IRREPS.num_irreps}x0e"))
        self.log_tau = nn.Parameter(torch.full((4,), float(torch.log(torch.tensor(tau0)))))      # one tau per order l = 0..3
        ls = []
        for l in range(4): ls += [l] * (4 * (2 * l + 1))
        self.register_buffer("lidx", torch.tensor(ls))
        self.tau_ch_l = [l for l in range(4) for _ in range(4)]                                   # per channel (16)
    def tau_comp(self): return self.log_tau.exp()[self.lidx]                                      # [64]
    def encode_g(self, feat):                      # global latent from pooled features: [B,1,64]
        return super().encode(feat=feat.mean(1, keepdim=True))
    def encode_d(self, feat):                      # cell deviations: [B,C,64]
        h = self.d2c(self.d1c(feat)); return 0.01 * self.mud(h), (0.1 * self.lvd(h, h) - 3.0).clamp(-12, 4)
    def kl_delta(self, md, lvd):
        lvx = self.expand_logvar(lvd); tau = self.tau_comp()
        return ((torch.log(tau) - 0.5 * lvx + (md ** 2 + lvx.exp()) / (2 * tau ** 2) - 0.5) * self.zmask).sum((-1, -2))
    def forward_stats(self, feat, stats, ncell, beta=1.0, alpha=0.0, sample=True):
        mug, lvg = self.encode_g(feat); md, lvd = self.encode_d(feat)
        zg = self.sample(mug, lvg) if sample else mug
        dl = self.sample(md, lvd) if sample else md
        lam = self.decode_cells(zg + dl)
        nll_cfg = (-(lam * stats).sum(-1) + ncell * self.logz(lam)).sum(-1)
        N = self.L ** 3
        pen = torch.tensor([l * (l + 1) for l in range(1, 7) for _ in range(2 * l + 1)], dtype=lam.dtype)
        reg = alpha * (lam ** 2 * pen).sum(-1).mean()
        kl_g = self.kl(mug, lvg); kl_d = self.kl_delta(md, lvd); kl = kl_g + kl_d
        return dict(loss=nll_cfg.mean() / N + beta * kl.mean() / N + reg, nll=nll_cfg.mean() / N, kl=kl.mean(), kl_g=kl_g.mean(), kl_d=kl_d.mean(),
                    lam=lam, mu=mug, lv=lvg, md=md, lvd=lvd)
    # evaluation helpers (posterior samples / means of the cell latent z_c)
    def cell_latent(self, feat, sample=True):
        mug, lvg = self.encode_g(feat); md, lvd = self.encode_d(feat)
        return (self.sample(mug, lvg) + self.sample(md, lvd)) if sample else (mug + md), (mug, lvg, md, lvd)
