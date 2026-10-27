"""E5 task A (nematic): linear response of the H=0 LL VAEs to (i) Zeeman l=1 field, (ii) quadrupolar field -eps sum P2(n.d), d = director of the sample.
chi = (beta/N)[E_z sum_i Var_i(O|z) + Var_z(sum_i <O_i>_z)]   (O = n_a n_b-type Zeeman component, or P2(n.d)); decoders are piecewise constant on R^3 cells of (12/R)^3 sites."""
import sys, os, math, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, f"{ROOT}/src"); os.chdir(f"{ROOT}/src")
from scipy.integrate import lebedev_rule
from mvae import sh
from mvae2 import cell_features
from mvae3 import MVAE3
from mvae8 import MVAE8
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
N = 1728
_X, _W = lebedev_rule(59); XQ = torch.tensor(_X.T, dtype=torch.float64); _W = torch.tensor(_W, dtype=torch.float64); LWQ = (_W * 4 * math.pi / _W.sum()).log()
YQ = sh(range(1, 7), XQ)                                                        # [Q,48]

def load_nem(path):
    ck = torch.load(path, weights_only=False); a = ck["args"]; R = a["R"]; cls = MVAE8 if a.get("abl") else MVAE3
    m = cls(L=12, R=R); m.load_state_dict(ck["state"], strict=False); m.eval(); m.T = a["T"]; m.R = R; m.cls = cls.__name__; m.args = a; return m
def probs(lam): return torch.softmax(lam.double() @ YQ.T + LWQ, -1)             # [...,Q]

@torch.no_grad()
def sample_prior(m, K, gen):
    C = m.R ** 3; zg = torch.randn(K, 1, 64, generator=gen) * m.zmask
    tau = m.tau_comp().detach(); z = zg + tau * torch.randn(K, C, 64, generator=gen)
    if m.cls == "MVAE8":                                                        # main l=2 channel: rot/amp deviations only, biaxial tied to z_g (subspaces from the z_g sample)
        a = m.act; tr, ta, _ = m.log_tau3.exp().detach(); Pr, Pa, Pb = m.projectors(zg[..., a:a + 5]); e = torch.randn(K, C, 5, generator=gen)
        z[..., a:a + 5] = zg[..., a:a + 5] + tr * torch.einsum("kij,kcj->kci", Pr[:, 0], e) + ta * torch.einsum("kij,kcj->kci", Pa[:, 0], e)
    return z * m.zmask
@torch.no_grad()
def posterior_z(m, Sv, gen):
    F = cell_features(Sv, m.R)
    if m.cls == "MVAE3": return m.cell_latent(F, sample=True)[0] * m.zmask
    mu_g, lv_g, la_g = m.encode3(F.mean(1, keepdim=True)); mu_c, lv_c, la_c = m.encode3(F); a = m.act
    Prot, Pamp, Pbi = m.projectors(mu_g[..., a:a + 5]); m5 = mu_c[..., a:a + 5]; sd = torch.exp(0.5 * la_c); eps = torch.randn_like(m5)
    zg = m.sample3(mu_g, lv_g, la_g)
    zm = torch.einsum("...ij,...j->...i", Prot + Pamp, m5) + sd[..., 0:1] * torch.einsum("...ij,...j->...i", Prot, eps) + sd[..., 1:2] * torch.einsum("...ij,...j->...i", Pamp, eps) + torch.einsum("...ij,...j->...i", Pbi, zg[..., a:a + 5])
    z = m.sample(mu_c, lv_c).clone(); z[..., a:a + 5] = zm; return z * m.zmask
@torch.no_grad()
def val_spins(T): S = torch.tensor(np.load(f"{ROOT}/data/ll_L12_T{T:.3f}_o.npy")); nv = max(len(S) // 10, 10); return S[-nv:]

@torch.no_grad()
def response(m, z, chunk=200):
    beta = 1 / m.T; mult = (12 // m.R) ** 3; res = []
    for i in range(0, len(z), chunk):
        lam = m.decode_cells(z[i:i + chunk]); p = probs(lam)                                         # [k,C,Q]
        m1 = p @ XQ; S2 = torch.einsum("kcq,qa,qb->kcab", p, XQ, XQ)                                 # site mean, second moment
        Ssum = mult * S2.sum(1); w, V = torch.linalg.eigh(Ssum); d = V[:, :, 2]; F = V[:, :, [2, 0, 1]]  # director first
        Cov = Ssum - mult * torch.einsum("kca,kcb->kab", m1, m1); Cf = torch.einsum("kia,kij,kjb->kab", F, Cov, F)
        Mf = torch.einsum("kia,ki->ka", F, mult * m1.sum(1))
        f = 1.5 * (d @ XQ.T) ** 2 - 0.5                                                              # [k,Q] P2(n.d(z))
        mq = torch.einsum("kcq,kq->kc", p, f); vq = torch.einsum("kcq,kq->kc", p, f * f) - mq ** 2
        res.append(torch.stack([0.5 * (Cf[:, 1, 1] + Cf[:, 2, 2]), Cf[:, 0, 0], 0.5 * (Mf[:, 1] ** 2 + Mf[:, 2] ** 2), Mf[:, 0] ** 2, mult * vq.sum(1), mult * mq.sum(1), (mult * mq.sum(1)) ** 2], 1))
    r = torch.cat(res); s = beta / N; mean = r.mean(0); se = lambda x: float(x.std() / math.sqrt(len(x)) * s)
    Sq = r[:, 5] / N; out = dict(zee_dec_perp=float(mean[0] * s), zee_dec_par=float(mean[1] * s), zee_lat_perp=float(mean[2] * s), zee_lat_par=float(mean[3] * s),
        quad_dec=float(mean[4] * s), quad_lat=float(r[:, 5].var() * s), quad_lat_alt=float(mean[6] * s - (r[:, 5].mean() ** 2) * s), S_mean=float(Sq.mean()),
        zee_dec_perp_se=se(r[:, 0]), zee_dec_par_se=se(r[:, 1]), quad_dec_se=se(r[:, 4]), quad_lat_se=se((r[:, 5] - r[:, 5].mean()) ** 2))
    out["spin_len"] = float(m1.norm(dim=-1).mean()); return out
