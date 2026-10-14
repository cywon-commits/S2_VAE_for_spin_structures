"""v3 2.3: equivariant posterior covariance for the (single) active l=2 latent channel.
The 5-vector mean mu is split by projectors built from mu itself (equivariant): amplitude (1) = mu_hat, rotation tangent (2) = top-2 singular
subspace of {J_k mu}, biaxial (2) = remainder; one variance per subspace.  Other (masked) channels are unused."""
import torch, torch.nn as nn
from e3nn import o3
from mvae2 import MVAE2

class MVAE4(MVAE2):
    def __init__(self, L=12, R=1, **kw):
        super().__init__(L=L, R=R, **kw)
        h = self.c2.gate.irreps_out
        self.lv3 = o3.FullyConnectedTensorProduct(h, h, o3.Irreps("3x0e")); self.b3 = nn.Parameter(torch.zeros(3))
        self.register_buffer("J", torch.stack(list(o3.so3_generators(2))).float())     # [3,5,5]
        self.register_buffer("act_idx", torch.tensor(-1))          # offset of the main active l=2 channel (in latent components)
    @property
    def act(self): return int(self.act_idx.item())
    def set_active(self, feat=None):
        """main active l=2 channel = the active channel with the largest mean |mu| (needs features if more than one is active)."""
        zm = self.zmask[16:36].reshape(4, 5); idx = [c for c in range(4) if zm[c].sum() > 0]; assert len(idx) >= 1
        if len(idx) > 1:
            with torch.no_grad(): mu, _ = MVAE2.encode(self, feat=feat)
            nrm = [mu[..., 16 + 5 * c:21 + 5 * c].norm(dim=-1).mean().item() for c in idx]; c = idx[int(torch.tensor(nrm).argmax())]
        else: c = idx[0]
        self.act_idx.fill_(16 + 5 * c); return c
    def encode3(self, feat):
        mu, lv = super().encode(feat=feat); h = self.c2(self.c1(feat)); c = (self.act - 16) // 5
        lv_a = lv[..., 8 + c:9 + c] + self.b3 + 0.1 * self.lv3(h, h)              # [...,3] = (rot, amp, biaxial) log-variances
        return mu, lv, lv_a.clamp(-12, 4)
    def projectors(self, mu5):
        """mu5 [...,5] -> P_rot, P_amp, P_bi  [...,5,5]  (detached)."""
        with torch.no_grad():
            m = mu5.double(); Jm = torch.einsum("kij,...j->...ik", self.J.double(), m)       # [...,5,3] columns J_k mu
            U, Sv, _ = torch.linalg.svd(Jm, full_matrices=False); Ur = U[..., :, :2]
            P_rot = Ur @ Ur.transpose(-1, -2); u = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-8); P_amp = u.unsqueeze(-1) * u.unsqueeze(-2)
            P_bi = torch.eye(5, dtype=torch.float64) - P_rot - P_amp
        return P_rot.float(), P_amp.float(), P_bi.float()
    def sample3(self, mu, lv, lv_a):
        z = self.sample(mu, lv)                                                       # other channels: scalar posterior variance
        eps = torch.randn_like(mu[..., self.act:self.act + 5]); m5 = mu[..., self.act:self.act + 5]; Pr, Pa, Pb = self.projectors(m5)
        sd = torch.exp(0.5 * lv_a)
        d = (torch.einsum("...ij,...j->...i", Pr, eps) * sd[..., 0:1] + torch.einsum("...ij,...j->...i", Pa, eps) * sd[..., 1:2] + torch.einsum("...ij,...j->...i", Pb, eps) * sd[..., 2:3])
        z = z.clone(); z[..., self.act:self.act + 5] = m5 + d; return z
    def kl3(self, mu, lv, lv_a):
        m5 = mu[..., self.act:self.act + 5]; dims = torch.tensor([2., 1., 2.])
        zm = self.zmask.clone(); zm[self.act:self.act + 5] = 0                          # other active channels: ordinary KL
        lvx = self.expand_logvar(lv); kl_other = (0.5 * (mu ** 2 + lvx.exp() - 1 - lvx) * zm).sum((-1, -2))
        return kl_other + 0.5 * ((m5 ** 2).sum(-1) + (dims * (lv_a.exp() - 1 - lv_a)).sum(-1)).sum(-1)
    def forward_stats(self, feat, stats, ncell, beta=1.0, alpha=0.0, sample=True):
        mu, lv, lv_a = self.encode3(feat); z = self.sample3(mu, lv, lv_a) if sample else mu * self.zmask
        lam = self.decode_cells(z); nll_cfg = (-(lam * stats).sum(-1) + ncell * self.logz(lam)).sum(-1); N = self.L ** 3
        pen = torch.tensor([l * (l + 1) for l in range(1, 7) for _ in range(2 * l + 1)], dtype=lam.dtype); reg = alpha * (lam ** 2 * pen).sum(-1).mean()
        kl = self.kl3(mu, lv, lv_a)
        return dict(loss=nll_cfg.mean() / N + beta * kl.mean() / N + reg, nll=nll_cfg.mean() / N, kl=kl.mean(), lam=lam, mu=mu, lv=lv, lv_a=lv_a)
