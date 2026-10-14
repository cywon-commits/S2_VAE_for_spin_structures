"""v2: cell-sufficient-statistic likelihood + feature encoder.
 log p(x|z) = sum_c [ lam_c(z) . sum_{i in c} Y(n_i) - n_c log Z(lam_c) ]   (decoder piecewise-constant per cell; R=1: one cell)
Encoder: fixed SO(3)-equivariant site features (gauge-invariant: even-l moments and nearest-neighbour bond products), mean-pooled per
cell, then an equivariant e3nn network on cells.  (v1's learnable site-level message passing cost ~2 s/step; infeasible for 25k+ steps.)"""
import torch, torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate
from mvae import MVAE, sh, LAT_IRREPS, LAM_IRREPS

FEAT_IRREPS = o3.Irreps("1x2e+1x4e+1x6e+1x0e+1x1e+1x2e+1x3e+1x4e")      # moments M2,M4,M6 ; bond products (2x2 -> 0..4)
_FTP = None
def _ftp():
    global _FTP
    if _FTP is None: _FTP = o3.FullTensorProduct(o3.Irreps("1x2e"), o3.Irreps("1x2e"))
    return _FTP

@torch.no_grad()
def cell_features(S, R, chunk=50):
    """S [C,L,L,L,3] -> [C,R^3,52]: per-cell means of Y_l(n_i) (l=2,4,6) and of Y_2(n_i) x Y_2(n_{i+d}) (d = x,y,z bonds averaged)."""
    C, L = S.shape[0], S.shape[1]; s = L // R; out = []; ftp = _ftp()
    for i in range(0, C, chunk):
        n = S[i:i + chunk].double(); B = len(n)
        Y2 = sh([2], n, "component"); Y4 = sh([4], n, "component"); Y6 = sh([6], n, "component")
        bond = 0
        ftp = ftp.double()
        for ax in (1, 2, 3): bond = bond + ftp(Y2, torch.roll(Y2, -1, dims=ax)) / 3
        f = torch.cat([Y2, Y4, Y6, bond], -1)                                   # [B,L,L,L,52]
        out.append(f.reshape(B, R, s, R, s, R, s, -1).mean((2, 4, 6)).reshape(B, R ** 3, -1).float())
    return torch.cat(out)

class FCLayer(nn.Module):
    def __init__(self, irr_in, c, lmax=3):
        super().__init__()
        sc = o3.Irreps(f"{c}x0e"); gated = o3.Irreps("+".join(f"{c}x{l}e" for l in range(1, lmax + 1)))
        self.gate = Gate(sc, [torch.nn.functional.silu], o3.Irreps(f"{gated.num_irreps}x0e"), [torch.sigmoid], gated)
        self.tp = o3.FullyConnectedTensorProduct(irr_in, irr_in, self.gate.irreps_in)
        self.lin = o3.Linear(irr_in, self.gate.irr_in if False else self.gate.irreps_in); self.skip = o3.Linear(irr_in, self.gate.irreps_out)
    def forward(self, x): return self.gate(self.tp(x, x) + self.lin(x)) + self.skip(x)

class MVAE2(MVAE):
    def __init__(self, L=12, R=1, cell_ch=8, dec_hid=4, order=59):
        super().__init__(L=L, R=R, hid=1, dec_hid=dec_hid, order=order)
        del self.l1, self.l2, self.mu, self.lv
        self.c1 = FCLayer(FEAT_IRREPS, cell_ch); self.c2 = FCLayer(self.c1.gate.irreps_out, cell_ch)
        h = self.c2.gate.irreps_out
        self.mu = o3.Linear(h, LAT_IRREPS); self.lv = o3.FullyConnectedTensorProduct(h, h, o3.Irreps(f"{LAT_IRREPS.num_irreps}x0e"))
        self.register_buffer("zmask", torch.ones(LAT_IRREPS.dim))      # 0 = pruned (posterior-collapsed) latent component
    def kl(self, mu, lv):
        lvx = self.expand_logvar(lv); return (0.5 * (mu ** 2 + lvx.exp() - 1 - lvx) * self.zmask).sum((-1, -2))
    def active_channels(self, mu, lv, thr=0.5):
        """mask over latent components: a channel is active if its mean KL (over configs and cells) exceeds thr nats."""
        lvx = self.expand_logvar(lv); klc = (0.5 * (mu ** 2 + lvx.exp() - 1 - lvx)).mean((0, 1)); mask = torch.zeros_like(klc); off = 0
        for l in range(4):
            for c in range(4):
                w = 2 * l + 1
                if klc[off:off + w].sum() > thr: mask[off:off + w] = 1
                off += w
        return mask
    def encode(self, n=None, feat=None):
        if feat is None: feat = cell_features(n.to(torch.get_default_dtype()) if n.dtype != torch.float64 else n, self.R).to(next(self.parameters()).dtype)
        h = self.c2(self.c1(feat))
        return 0.1 * self.mu(h), (0.1 * self.lv(h, h) - 3.0).clamp(-12, 4)
    def decode_cells(self, z):                      # piecewise constant per cell: z [B,C,64] -> lam [B,C,48]
        z = z * self.zmask
        return 0.01 * (self.d2(self.d1(z, z), z) + self.dlin(z))
    def forward_stats(self, feat, stats, ncell, beta=1.0, alpha=0.0, sample=True):
        """feat [B,C,52]; stats [B,C,48] = sum_{i in cell} Y_l(n_i) (integral norm, l=1..6); ncell = sites per cell."""
        mu, lv = self.encode(feat=feat); z = self.sample(mu, lv) if sample else mu
        lam = self.decode_cells(z)
        nll_cfg = (-(lam * stats).sum(-1) + ncell * self.logz(lam)).sum(-1)       # [B] total NLL of the configuration
        N = self.L ** 3
        pen = torch.tensor([l * (l + 1) for l in range(1, 7) for _ in range(2 * l + 1)], dtype=lam.dtype)
        reg = alpha * (lam ** 2 * pen).sum(-1).mean()
        kl = self.kl(mu, lv)
        return dict(loss=nll_cfg.mean() / N + beta * kl.mean() / N + reg, nll=nll_cfg.mean() / N, kl=kl.mean(), lam=lam, mu=mu, lv=lv)

@torch.no_grad()
def compute_stats(S, R, chunk=50):
    """S [C,L,L,L,3] -> stats [C,R^3,48] (cell sums of integral-normalised Y_{1..6})."""
    C, L = S.shape[0], S.shape[1]; s = L // R; out = []
    for i in range(0, C, chunk):
        Y = sh(range(1, 7), S[i:i + chunk].double()).float()
        out.append(Y.reshape(len(Y), R, s, R, s, R, s, 48).sum((2, 4, 6)).reshape(len(Y), R ** 3, 48))
    return torch.cat(out)
