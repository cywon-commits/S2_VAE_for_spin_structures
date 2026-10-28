"""E6 task A: channel-resolved linear response of the nematic VAEs: m=0 (P2(n.d)) and m=+-2 (B=(n.e1)^2-(n.e2)^2, C=2(n.e1)(n.e2)).
Frame per z: d = top eigvec of sum_i <n n^T>; e1 = lab-x projected on plane perp d; e2 = d x e1  (same rule as fdt_biax).
chi_B = (beta/N) * 1/2 * [ E_z sum_i (Var B_i + Var C_i) + Var_z(sum<B>) + Var_z(sum<C>) ] (covariances over z: mean subtracted; mean reported)."""
from nem_response import *
from e3nn import o3
J = torch.stack(list(o3.so3_generators(2))).float()
def main_channel(m):
    """offset of the main active l=2 channel (MVAE8: m.act; MVAE3: active l=2 channel with largest mean |z| later -> chosen from zmask only if unique, else set by caller)."""
    return m.act if hasattr(m, "act") else getattr(m, "_act", None)
def projectors(m5):
    mm = m5.double(); Jm = torch.einsum("kij,...j->...ik", J.double(), mm); U, S, _ = torch.linalg.svd(Jm, full_matrices=False); Ur = U[..., :, :2]
    Pr = Ur @ Ur.transpose(-1, -2); u = mm / mm.norm(dim=-1, keepdim=True).clamp_min(1e-8); Pa = u.unsqueeze(-1) * u.unsqueeze(-2); Pb = torch.eye(5, dtype=torch.float64) - Pr - Pa
    return Pr.float(), Pa.float(), Pb.float()
@torch.no_grad()
def set_main(m, z):
    if hasattr(m, "act"): return m.act
    zm = m.zmask[16:36].reshape(4, 5); idx = [c for c in range(4) if zm[c].sum() > 0]; nrm = [z[..., 16 + 5 * c:21 + 5 * c].norm(dim=-1).mean().item() for c in idx]; m._act = 16 + 5 * idx[int(np.argmax(nrm))]; return m._act
@torch.no_grad()
def response_ch(m, z, chunk=200):
    beta = 1 / m.T; mult = (12 // m.R) ** 3; rows = []; xh = torch.tensor([1.0, 0, 0], dtype=torch.float64); yh = torch.tensor([0, 1.0, 0], dtype=torch.float64)
    for i in range(0, len(z), chunk):
        lam = m.decode_cells(z[i:i + chunk]); p = probs(lam); k = len(p)
        S2 = torch.einsum("kcq,qa,qb->kcab", p, XQ, XQ); Ssum = mult * S2.sum(1); w, V = torch.linalg.eigh(Ssum); d = V[:, :, 2]
        e1 = xh - (d @ xh)[:, None] * d; bad = e1.norm(dim=-1) < 1e-3; e1 = torch.where(bad[:, None], yh - (d @ yh)[:, None] * d, e1); e1 = e1 / e1.norm(dim=-1, keepdim=True); e2 = torch.cross(d, e1, dim=-1)
        a, b = XQ @ e1.T, XQ @ e2.T                                                                # [Q,k]
        fq = 1.5 * (XQ @ d.T) ** 2 - 0.5; fB = a ** 2 - b ** 2; fC = 2 * a * b
        out = []
        for f in (fq, fB, fC):
            f = f.T; mu = torch.einsum("kcq,kq->kc", p, f); var = torch.einsum("kcq,kq->kc", p, f * f) - mu ** 2; out += [mult * var.sum(1), mult * mu.sum(1)]
        rows.append(torch.stack(out, 1))
    r = torch.cat(rows); s = beta / N                                                              # cols: decQ, MQ, decB, MB, decC, MC
    res = dict(Q_dec=float(r[:, 0].mean() * s), Q_lat=float(r[:, 1].var() * s), B_dec=float(0.5 * (r[:, 2] + r[:, 4]).mean() * s), B_lat=float(0.5 * (r[:, 3].var() + r[:, 5].var()) * s),
               B_mean=float(r[:, 3].mean() / N), C_mean=float(r[:, 5].mean() / N), B_lat_nomean=float(0.5 * ((r[:, 3] ** 2).mean() + (r[:, 5] ** 2).mean()) * s))
    res["B_lat_se"] = float(0.5 * ((r[:, 3] - r[:, 3].mean()) ** 2 + (r[:, 5] - r[:, 5].mean()) ** 2).std() / math.sqrt(len(r)) * s); return res
@torch.no_grad()
def split_deviations(m, z, keep):
    """keep in {'none','bi','rot','rotamp','all'}: z_g := mean over cells; main-channel deviation delta=z_c - z_g projected onto the kept subspace (projectors from z_g main 5-vector); all other-channel deviations zeroed (except keep=='all')."""
    if keep == "all": return z
    a = set_main(m, z); zg = z.mean(1, keepdim=True); dz = z - zg; out = zg.expand_as(z).clone(); Pr, Pa, Pb = projectors(zg[..., a:a + 5]); d5 = dz[..., a:a + 5]
    P = {"none": None, "bi": Pb, "rot": Pr, "rotamp": Pr + Pa}[keep]
    if P is not None: out[..., a:a + 5] = out[..., a:a + 5] + torch.einsum("kij,kcj->kci", P[:, 0], d5)
    return out * m.zmask
