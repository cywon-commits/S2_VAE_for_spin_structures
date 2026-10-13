"""SO(3)-equivariant VAE with a general multipole (exponential-family) decoder for S^2 spin fields on an LxLxL lattice.
SO(3) acts on the internal spin space only (global rotation of every spin); the lattice is not rotated."""
import math, numpy as np, torch, torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate
from scipy.integrate import lebedev_rule

LMAX_DEC = 6
LAT_IRREPS = o3.Irreps("4x0e+4x1e+4x2e+4x3e")      # latent channels l=0..3, 4 each
LAM_IRREPS = o3.Irreps("1x1e+1x2e+1x3e+1x4e+1x5e+1x6e")
OFFS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def sh(l_list, n, norm="integral"):
    """real SH (e3nn basis) of unit vectors n [...,3], concatenated for l in l_list."""
    return torch.cat([o3.spherical_harmonics(l, n, normalize=True, normalization=norm) for l in l_list], -1)


class LogZ(nn.Module):
    """log Z(lam) = log int exp(lam.Y(n)) dOmega via Lebedev quadrature, l=1..6 (integral-normalised Y)."""
    def __init__(self, order=59):
        super().__init__()
        X, W = lebedev_rule(order)
        X = torch.tensor(X.T, dtype=torch.float64); W = torch.tensor(W, dtype=torch.float64)
        W = W * 4 * math.pi / W.sum()
        self.register_buffer("Yq", sh(range(1, LMAX_DEC + 1), X))
        self.register_buffer("logw", W.log())
    def forward(self, lam):
        Yq = self.Yq.to(lam.dtype)
        return torch.logsumexp(lam @ Yq.T + self.logw.to(lam.dtype), -1)


def tp_gate(irr_in1, irr_in2, c, lmax=3):
    """channel-wise ('uuu') tensor product of two same-multiplicity irreps -> Linear -> gate-ready irreps."""
    sc = o3.Irreps(f"{c}x0e"); gated = o3.Irreps("+".join(f"{c}x{l}e" for l in range(1, lmax + 1)))
    gates = o3.Irreps(f"{gated.num_irreps}x0e")
    gate = Gate(sc, [torch.nn.functional.silu], gates, [torch.sigmoid], gated)
    return DepthwiseTP(irr_in1, irr_in2, gate.irreps_in), gate


class DepthwiseTP(nn.Module):
    """x (irreps with multiplicity c per l), y (same multiplicities, any l set): channel-wise product over all allowed paths,
    then equivariant Linear to irr_out. Much cheaper than FullyConnectedTensorProduct, still SO(3)-equivariant."""
    def __init__(self, i1, i2, irr_out, lmax=3):
        super().__init__()
        self.i1, self.i2 = o3.Irreps(i1), o3.Irreps(i2)
        outs, ins = [], []
        for a, (m1, ir1) in enumerate(self.i1):
            for b, (m2, ir2) in enumerate(self.i2):
                if m1 != m2: continue
                for ir3 in ir1 * ir2:
                    if ir3.l > lmax: continue
                    outs.append((m1, ir3)); ins.append((a, b, len(outs) - 1, "uuu", False))
        self.tp = o3.TensorProduct(self.i1, self.i2, o3.Irreps(outs), ins, shared_weights=True, internal_weights=False)
        self.lin = o3.Linear(self.tp.irreps_out, irr_out)
    def forward(self, x, y):
        return self.lin(self.tp(x, y, torch.ones(self.tp.weight_numel, dtype=x.dtype, device=x.device)))


class MPLayer(nn.Module):
    """m_i = sum_d Lin_d(x_{i+d}) + Lin_0(x_i);  y_i = Gate(TP(m_i, x_i)) + Lin_skip(x_i)."""
    def __init__(self, irr_in, c, lmax=3):
        super().__init__()
        gated = o3.Irreps("+".join([f"{c}x0e"] + [f"{c}x{l}e" for l in range(1, lmax + 1)]))
        self.irr_out = gated
        self.lin_nb = nn.ModuleList([o3.Linear(irr_in, gated) for _ in OFFS])
        self.lin_self = o3.Linear(irr_in, gated)
        self.lin_x = o3.Linear(irr_in, gated)
        self.tp, self.gate = tp_gate(gated, gated, c, lmax)
        self.skip = o3.Linear(irr_in, self.gate.irreps_out)
    def forward(self, x):                      # x [B,L,L,L,C]
        m = self.lin_self(x)
        for lin, d in zip(self.lin_nb, OFFS):
            m = m + lin(torch.roll(x, shifts=(-d[0], -d[1], -d[2]), dims=(1, 2, 3)))
        return self.gate(self.tp(m, self.lin_x(x))) + self.skip(x)


def interp_weights(L, R):
    """periodic trilinear interpolation matrix (L sites <- R cell centres) along one axis, [L,R]."""
    s = L / R; M = np.zeros((L, R))
    for i in range(L):
        u = (i + 0.5) / s - 0.5; i0 = math.floor(u); f = u - i0
        M[i, i0 % R] += 1 - f; M[i, (i0 + 1) % R] += f
    return torch.tensor(M, dtype=torch.float32)


class MVAE(nn.Module):
    def __init__(self, L=12, R=1, hid=3, dec_hid=4, order=59, Lcut=6):
        super().__init__()
        self.L, self.R, self.Lcut = L, R, Lcut
        irr_in = o3.Irreps("1x0e+1x1e+1x2e")
        self.l1 = MPLayer(irr_in, hid); self.l2 = MPLayer(self.l1.gate.irreps_out, hid)
        h = self.l2.gate.irreps_out
        self.mu = o3.Linear(h, LAT_IRREPS)
        self.lv = o3.FullyConnectedTensorProduct(h, h, o3.Irreps(f"{LAT_IRREPS.num_irreps}x0e"))  # one log-variance per latent channel
        self.n_ch = LAT_IRREPS.num_irreps
        self.ch_slices = [(sl.start, sl.stop) for sl in LAT_IRREPS.slices()]
        # decoder: z -> lam_{l=1..6}
        hid_dec = o3.Irreps("+".join(f"{dec_hid}x{l}e" for l in range(0, 7)))
        self.d1 = o3.FullyConnectedTensorProduct(LAT_IRREPS, LAT_IRREPS, hid_dec)
        self.d2 = o3.FullyConnectedTensorProduct(hid_dec, LAT_IRREPS, LAM_IRREPS)
        self.dlin = o3.Linear(LAT_IRREPS, LAM_IRREPS)
        if R > 1:
            M = interp_weights(L, R); self.register_buffer("M", M)
        self.logz = LogZ(order)
        self.lam_sl = [(sl.start, sl.stop) for sl in LAM_IRREPS.slices()]   # l=1..6
        self.lam_l = [l for l in range(1, 7)]
        self.register_buffer("lmask", torch.ones(48))
        self.lmask_set = 6
    # --- encoder ---
    def features(self, n):                     # n [B,L,L,L,3] unit spins
        B = n.shape[0]
        return torch.cat([torch.ones_like(n[..., :1]), sh([1, 2], n, "component")], -1)
    def encode(self, n):
        x = self.l2(self.l1(self.features(n)))
        R = self.R; L = self.L; B = n.shape[0]; s = L // R
        xc = x.reshape(B, R, s, R, s, R, s, -1).mean((2, 4, 6)).reshape(B, R ** 3, -1)
        return 0.1 * self.mu(xc), (0.1 * self.lv(xc, xc) - 3.0).clamp(-12, 4)   # mu [B,R^3,64]; logvar [B,R^3,16] (scalar per channel)
    def expand_logvar(self, lv):
        parts = [lv[..., i:i + 1].expand(*lv.shape[:-1], 2 * l + 1) for i, l in enumerate([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)]
        return torch.cat(parts, -1)
    def sample(self, mu, lv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * self.expand_logvar(lv))
    def kl(self, mu, lv):                      # per config, sum over cells and components
        lvx = self.expand_logvar(lv)
        return 0.5 * (mu ** 2 + lvx.exp() - 1 - lvx).sum((-1, -2))
    # --- decoder ---
    def decode(self, z, site_idx=None):        # z [B,R^3,64] -> lam [B,S,48]  (S=1 for global latent; S=len(site_idx) or L^3 otherwise)
        B = z.shape[0]
        if self.R > 1:
            L = self.L
            zg = z.reshape(B, self.R, self.R, self.R, -1)
            if site_idx is None:
                zg = torch.einsum("ia,jb,kc,zabcy->zijky", self.M, self.M, self.M, zg)
                z = zg.reshape(B, L ** 3, -1)
            else:
                i, j, k = site_idx // (L * L), (site_idx // L) % L, site_idx % L
                z = torch.einsum("sa,sb,sc,zabcy->zsy", self.M[i], self.M[j], self.M[k], zg)
        h = self.d1(z, z)
        return 0.01 * (self.d2(h, z) + self.dlin(z))      # small initial multipole amplitudes
    def lam_masked(self, lam, Lc):
        if Lc >= 6: return lam
        return lam[..., : sum(2 * l + 1 for l in range(1, Lc + 1))]
    def nll_sites(self, lam, n, Lc=6):        # lam [B,N or 1,48], n [B,N,3] -> per-site NLL [B,N]
        Y = sh(range(1, Lc + 1), n)
        lamc = self.lam_masked(lam, Lc)
        lz = self.logz(lam) if Lc >= 6 else self._logz_cut(lamc, Lc)       # [B,N] or [B,1]
        return -(lamc * Y).sum(-1) + lz
    def _logz_cut(self, lamc, Lc):
        k = lamc.shape[-1]; Yq = self.logz.Yq[:, :k].to(lamc.dtype)
        return torch.logsumexp(lamc @ Yq.T + self.logz.logw.to(lamc.dtype), -1)
    def forward(self, n, beta=1.0, alpha=0.0, Lc=6, sample=True, n_sub=None):
        B = n.shape[0]
        mu, lv = self.encode(n)
        z = self.sample(mu, lv) if sample else mu
        nflat = n.reshape(B, -1, 3)
        site_idx = None
        if n_sub is not None and self.R > 1:       # unbiased site subsample (same sites for the whole batch)
            site_idx = torch.randperm(nflat.shape[1])[:n_sub]; nflat = nflat[:, site_idx]
        lam = self.decode(z, site_idx)
        nll = self.nll_sites(lam, nflat, Lc)               # lam [B,1,48] broadcasts over sites for global latent
        pen = torch.tensor([l * (l + 1) for l in range(1, 7) for _ in range(2 * l + 1)], dtype=lam.dtype)
        reg = alpha * (self.lam_masked(lam, Lc) ** 2 * pen[: self.lam_masked(lam, Lc).shape[-1]]).sum(-1).mean()
        kl = self.kl(mu, lv)
        n_sites = self.L ** 3
        loss = nll.mean() + beta * kl.mean() / n_sites + reg   # per-site normalised ELBO
        return dict(loss=loss, nll=nll.mean(), kl=kl.mean(), lam=lam, mu=mu, lv=lv)
