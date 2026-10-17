"""E3: kagome Heisenberg AF.  SO(3)-equivariant VAE with per-cell, per-sublattice multipole decoder (l=1..8), cell sufficient-statistic likelihood.
Spin array layout: S[config, x, y, sublattice(0..2), 3].  Cells = (L/R)x(L/R) unit cells (3 (L/R)^2 sites)."""
import math, numpy as np, torch, torch.nn as nn
from e3nn import o3
from scipy.integrate import lebedev_rule
import sys; sys.path.insert(0, "..")
from mvae import sh
from mvae2 import FCLayer

LAT = o3.Irreps("4x0e+4x1e+4x2e+4x3e")
LMAX = 8
OUT = o3.Irreps("+".join(f"3x{l}e" for l in range(1, LMAX + 1)))          # 3 sublattices per l
FEAT = o3.Irreps("3x1e+3x2e+3x3e+3x4e+6x0e+6x1e+6x2e")                    # sublattice means of Y_l (l=1..4) ; nearest-neighbour bond products Y1 x Y1
BONDS = [(0, 1, 0, 0), (0, 1, -1, 0), (0, 2, 0, 0), (0, 2, 0, -1), (1, 2, 0, 0), (1, 2, 1, -1)]   # (sub_i, sub_j, dx, dy): j at cell + (dx,dy)
NY = sum(2 * l + 1 for l in range(1, LMAX + 1))                            # 80

_FTP = None
@torch.no_grad()
def kag_features(S, R, chunk=20):
    """S [C,L,L,3,3] -> [C,R^2,126] per-cell means (SO(3)-equivariant features)."""
    global _FTP
    if _FTP is None: _FTP = o3.FullTensorProduct(o3.Irreps("1x1e"), o3.Irreps("1x1e")).double()
    C, L = S.shape[0], S.shape[1]; b = L // R; out = []
    for i in range(0, C, chunk):
        n = S[i:i + chunk].double(); B = len(n); parts = []
        for l in (1, 2, 3, 4):
            Y = sh([l], n, "component"); parts.append(Y.reshape(B, R, b, R, b, 3, -1).mean((2, 4)).reshape(B, R * R, -1))
        bd = []
        for si, sj, dx, dy in BONDS:
            ni = n[:, :, :, si]; nj = torch.roll(n[:, :, :, sj], (-dx, -dy), dims=(1, 2))
            P = _FTP(sh([1], ni, "component"), sh([1], nj, "component"))            # [B,L,L,9] = 0e+1e+2e
            bd.append(P.reshape(B, R, b, R, b, 9).mean((2, 4)))                      # [B,R,R,9]
        bd = torch.stack(bd, 3).reshape(B, R * R, 6, 9)
        for lo, hi in ((0, 1), (1, 4), (4, 9)): parts.append(bd[..., lo:hi].reshape(B, R * R, -1))
        out.append(torch.cat(parts, -1).float())
    return torch.cat(out)

@torch.no_grad()
def kag_stats(S, R, chunk=20):
    """S -> [C,R^2,3,80]: sums over the sites of each (cell, sublattice) of integral-normalised Y_l (l=1..8)."""
    C, L = S.shape[0], S.shape[1]; b = L // R; out = []
    for i in range(0, C, chunk):
        Y = sh(range(1, LMAX + 1), S[i:i + chunk].double()).float(); B = len(Y)
        out.append(Y.reshape(B, R, b, R, b, 3, NY).sum((2, 4)).reshape(B, R * R, 3, NY))
    return torch.cat(out)

class LogZ(nn.Module):
    def __init__(self, order=59):
        super().__init__()
        X, W = lebedev_rule(order); X = torch.tensor(X.T, dtype=torch.float64); W = torch.tensor(W, dtype=torch.float64); W = W * 4 * math.pi / W.sum()
        self.register_buffer("Yq", sh(range(1, LMAX + 1), X)); self.register_buffer("logw", W.log())
    def forward(self, lam): return torch.logsumexp(lam @ self.Yq.to(lam.dtype).T + self.logw.to(lam.dtype), -1)

def to_sub(out):
    """e3nn layout of 3x{l}e blocks ([3,2l+1] per l) -> [...,3,80]."""
    parts = []; off = 0
    for l in range(1, LMAX + 1):
        w = 3 * (2 * l + 1); parts.append(out[..., off:off + w].reshape(*out.shape[:-1], 3, 2 * l + 1)); off += w
    return torch.cat(parts, -1)

class KVAE(nn.Module):
    def __init__(self, R=1, cell_ch=8, dec_hid=4, order=59):
        super().__init__()
        self.R, self.L = R, 12
        self.c1 = FCLayer(FEAT, cell_ch); self.c2 = FCLayer(self.c1.gate.irreps_out, cell_ch); h = self.c2.gate.irreps_out
        self.mu = o3.Linear(h, LAT); self.lv = o3.FullyConnectedTensorProduct(h, h, o3.Irreps(f"{LAT.num_irreps}x0e"))
        hid = o3.Irreps("+".join(f"{dec_hid}x{l}e" for l in range(0, 7)))
        self.d1 = o3.FullyConnectedTensorProduct(LAT, LAT, hid); self.d2 = o3.FullyConnectedTensorProduct(hid, LAT, OUT); self.dlin = o3.Linear(LAT, OUT)
        self.logz = LogZ(order); self.register_buffer("zmask", torch.ones(LAT.dim))
        self.register_buffer("pen", torch.tensor([l * (l + 1) for l in range(1, LMAX + 1) for _ in range(2 * l + 1)], dtype=torch.float32))
    def encode(self, feat):
        h = self.c2(self.c1(feat)); return 0.1 * self.mu(h), (0.1 * self.lv(h, h) - 3.0).clamp(-12, 4)
    def expand_logvar(self, lv):
        parts = [lv[..., i:i + 1].expand(*lv.shape[:-1], 2 * l + 1) for i, l in enumerate([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)]; return torch.cat(parts, -1)
    def sample(self, mu, lv): return mu + torch.randn_like(mu) * torch.exp(0.5 * self.expand_logvar(lv))
    def kl(self, mu, lv):
        lvx = self.expand_logvar(lv); return (0.5 * (mu ** 2 + lvx.exp() - 1 - lvx) * self.zmask).sum((-1, -2))
    def decode(self, z):
        z = z * self.zmask; return to_sub(0.01 * (self.d2(self.d1(z, z), z) + self.dlin(z)))      # [B,C,3,80]
    def nll_cfg(self, lam, stats, nsub):
        return (-(lam * stats).sum(-1) + nsub * self.logz(lam)).sum((-1, -2))                      # [B]
    def forward_stats(self, feat, stats, nsub, beta=1.0, alpha=0.0, sample=True):
        mu, lv = self.encode(feat); z = self.sample(mu, lv) if sample else mu; lam = self.decode(z)
        nll = self.nll_cfg(lam, stats, nsub); N = 3 * self.L ** 2; kl = self.kl(mu, lv); reg = alpha * (lam ** 2 * self.pen).sum(-1).mean()
        return dict(loss=nll.mean() / N + beta * kl.mean() / N + reg, nll=nll.mean() / N, kl=kl.mean(), lam=lam, mu=mu, lv=lv)
