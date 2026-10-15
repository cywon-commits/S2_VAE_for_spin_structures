"""v4 task B: separate direction and magnitude in the decoder.
The main active l=2 latent channel z_2 (5 comps) is replaced by its unit vector z_hat in the decoder input and its norm r enters only through
rotation-invariant scalars phi(r) (small MLP) placed in the l=0 slots, so lam_2 = g(r) z_hat is representable and the encoder is free to choose |mu|."""
import torch, torch.nn as nn
from mvae2 import MVAE2

class MVAE6(MVAE2):
    def __init__(self, L=12, R=1, **kw):
        super().__init__(L=L, R=R, **kw)
        self.phi = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 2))
        self.register_buffer("act_idx", torch.tensor(-1))
    @property
    def act(self): return int(self.act_idx.item())
    def set_active(self, feat=None):
        zm = self.zmask[16:36].reshape(4, 5); idx = [c for c in range(4) if zm[c].sum() > 0]; assert len(idx) >= 1
        if len(idx) > 1:
            with torch.no_grad(): mu, _ = MVAE2.encode(self, feat=feat)
            nrm = [mu[..., 16 + 5 * c:21 + 5 * c].norm(dim=-1).mean().item() for c in idx]; c = idx[int(torch.tensor(nrm).argmax())]
        else: c = idx[0]
        self.act_idx.fill_(16 + 5 * c); return c
    def active_l2(self):
        zm = self.zmask[16:36].reshape(4, 5); return [16 + 5 * c for c in range(4) if zm[c].sum() > 0]
    def decode_cells(self, z):
        """separate direction and magnitude for EVERY active l=2 channel (the optimiser moved the information to the unseparated channel when only
        the main channel was separated): channel -> unit vector, norms -> scalars phi(r) in the l=0 slots (2 slots per channel, up to 2 channels)."""
        z = z * self.zmask; zin = z.clone(); sc = []
        for off in self.active_l2():
            zm = z[..., off:off + 5]; r = zm.norm(dim=-1, keepdim=True); zin[..., off:off + 5] = zm / r.clamp_min(1e-6); sc.append(self.phi(r))
        sc = torch.cat(sc, -1)[..., :4]; zin[..., 0:sc.shape[-1]] = sc
        return 0.01 * (self.d2(self.d1(zin, zin), zin) + self.dlin(zin))
    def reinit_decoder(self):
        for mod in (self.d1, self.d2, self.dlin):
            for p in mod.parameters(): nn.init.normal_(p, std=1.0) if p.dim() > 0 else None
