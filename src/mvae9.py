"""v5 1.4: task B + task A.  Subspace-covariance up/down model (MVAE7) with the direction/magnitude-separated decoder of MVAE6
(every active l=2 channel -> unit vector, norms -> scalars phi(r) in the l=0 slots)."""
import torch, torch.nn as nn
from mvae7 import MVAE7

class MVAE9(MVAE7):
    def __init__(self, L=12, R=2, **kw):
        super().__init__(L=L, R=R, **kw)
        self.phi = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 2))
    def active_l2(self):
        zm = self.zmask[16:36].reshape(4, 5); return [16 + 5 * c for c in range(4) if zm[c].sum() > 0]
    def decode_cells(self, z):
        z = z * self.zmask; zin = z.clone(); sc = []
        for off in self.active_l2():
            zm = z[..., off:off + 5]; r = zm.norm(dim=-1, keepdim=True); zin[..., off:off + 5] = zm / r.clamp_min(1e-6); sc.append(self.phi(r))
        sc = torch.cat(sc, -1)[..., :4]; zin[..., 0:sc.shape[-1]] = sc
        return 0.01 * (self.d2(self.d1(zin, zin), zin) + self.dlin(zin))
