"""(a) equivariance test: lam(R n) == D(R) lam(n) (double precision, posterior mean)."""
import torch, numpy as np, sys
from e3nn import o3
from mvae import MVAE, LAM_IRREPS
torch.manual_seed(0)
def check(R, L=12):
    m = MVAE(L=L, R=R).double().eval()
    n = torch.randn(2, L, L, L, 3, dtype=torch.float64); n = n / n.norm(dim=-1, keepdim=True)
    rot = o3.rand_matrix(dtype=torch.float64)
    with torch.no_grad():
        mu, lv = m.encode(n); lam = m.decode(mu)
        mu2, lv2 = m.encode(n @ rot.T); lam2 = m.decode(mu2)
        D = LAM_IRREPS.D_from_matrix(rot)
    err = ((lam2 - lam @ D.T).norm() / lam.norm()).item()
    errlv = (lv2 - lv).abs().max().item()
    print(f"R={R}: rel err lam = {err:.2e}, logvar invariance = {errlv:.2e}"); return err
def check_trained(path, n_rot=3):
    """(a) on a trained checkpoint with real MC spin configs: lam(R n) vs D(R) lam(n), double precision, posterior mean z."""
    import numpy as np
    from evaluate import load, val_data
    m, a = load(path); m = m.double(); S = val_data(a)[:4].double(); worst = 0.
    for _ in range(n_rot):
        rot = o3.rand_matrix(dtype=torch.float64)
        with torch.no_grad():
            lam = m.decode(m.encode(S)[0]); lam2 = m.decode(m.encode(S @ rot.T)[0]); D = LAM_IRREPS.D_from_matrix(rot)
        worst = max(worst, ((lam2 - lam @ D.T).norm() / lam.norm()).item())
    return worst

if __name__ == "__main__":
    errs = [check(R) for R in (1, 2, 3, 4)]
    assert max(errs) < 1e-5

