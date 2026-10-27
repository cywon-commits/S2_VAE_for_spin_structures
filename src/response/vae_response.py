"""E5: linear and non-linear response of H=0-trained kagome VAEs to an external (Zeeman) field.
Site distributions are exponential families p_i(n) ∝ exp(lambda_i . Y(n)); a Zeeman term beta*h*n.H adds beta*h*(4pi/3)*Y_1m(H) to lambda_i's l=1 part, so site moments follow exactly from Lebedev quadrature.
Conventions: lam layout [..., 80] = l=1..8 (e3nn real basis, integral normalisation, as everywhere in src/kag); N = 432 sites; beta = 1/T (J=1).
    chi_ab = (beta/N) [ E_z sum_i Cov_i(n_a, n_b | z)  +  E_z Mbar_a(z) Mbar_b(z) ],   Mbar(z) = sum_i <n_i>_z      (sites independent given z; <M>=0 by isotropy)
in the sample frame of each z: nu(z) = lowest eigenvector of sum_i <n_i n_i^T>_z (plane normal), e1,e2 in plane (in-plane value = mean of the two)."""
import sys, os, math, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
for p in ("src", "src/kag", "src/kagome", "src/layered"): sys.path.insert(0, f"{ROOT}/{p}")
from kmodel import KVAE, LAT, LMAX, kag_features, kag_stats
from kmodel5 import KVAE5
from kmodel6 import KVAE6, site_Y
from kmodel6u import KVAE6U
from eval_k import XQ, YQ, LW                                   # Lebedev order 131 (double): points [Q,3], Y basis [Q,80], log weights [Q]
from mvae import sh
from kdata import load_T, VAL_SEEDS, TRAIN_SEEDS
N_SITES = 432; XX9 = (XQ[:, :, None] * XQ[:, None, :]).reshape(len(XQ), 9)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

def load_kag(path):
    ck = torch.load(path, weights_only=False); a = ck["args"]; name = os.path.basename(path)
    R = a["R"]; state = {k: v for k, v in ck["state"].items() if not k.startswith("logz.")}
    if name.startswith("kag6u_"): m = KVAE6U(R=R)
    elif name.startswith("kag6_"): m = KVAE6(R=R)
    elif name.startswith("kagu_"): m = KVAE5(R=R)
    else: m = KVAE(R=R)
    m.load_state_dict(state, strict=False); m.eval(); m.kind = ("6u" if name.startswith("kag6u_") else "6" if name.startswith("kag6_") else "5" if name.startswith("kagu_") else "1"); m.T = a["T"]; m.R = R; m.args = a
    return m

@torch.no_grad()
def site_moments(lam, chunk=3000):
    """lam [...,80] -> mean [...,3], second moment [...,3,3] of p ∝ exp(lam.Y(n)) (Lebedev 131)."""
    sh_ = lam.shape[:-1]; flat = lam.reshape(-1, lam.shape[-1]).double(); M = torch.empty(len(flat), 3, dtype=torch.float64); S = torch.empty(len(flat), 9, dtype=torch.float64)
    for i in range(0, len(flat), chunk):
        p = torch.softmax(flat[i:i + chunk] @ YQ.T + LW, -1); M[i:i + chunk] = p @ XQ; S[i:i + chunk] = p @ XX9
    return M.reshape(*sh_, 3), S.reshape(*sh_, 3, 3)

def zeeman_shift(hvec, beta):
    """delta lambda [..., 80] (only l=1 block non-zero) for the energy term -h.n (weight exp(+beta h.n)): beta*|h|*(4pi/3)*Y_1m(hhat), integral-normalised Y."""
    hvec = torch.as_tensor(hvec, dtype=torch.float64); nrm = hvec.norm(dim=-1, keepdim=True).clamp_min(1e-300); Y1 = sh([1], hvec / nrm)                    # [...,3]
    d = torch.zeros(*hvec.shape[:-1], LMAX * (LMAX + 2), dtype=torch.float64); d[..., :3] = beta * nrm * (4 * math.pi / 3) * Y1; return d

def log_z(lam, chunk=3000):
    sh_ = lam.shape[:-1]; flat = lam.reshape(-1, lam.shape[-1]).double(); out = torch.empty(len(flat), dtype=torch.float64)
    with torch.no_grad():
        for i in range(0, len(flat), chunk): out[i:i + chunk] = torch.logsumexp(flat[i:i + chunk] @ YQ.T + LW, -1)
    return out.reshape(sh_)

# ------------------------------------------------------------------ latent sampling and decoding
def mult_of(m): return 1.0 if m.kind in ("6", "6u") else float((12 // m.R) ** 2)          # number of unit cells sharing one (cell, sublattice) lambda
def sample_prior(m, K, gen):
    """ancestral samples of the latent fed to the decoder: kind 1: z~N(0,I); kinds 5, 6, 6u: z_g~N(0,I), z_c = z_g + tau_l*eps per cell.  Returns z [K,C,64]."""
    C = m.R ** 2; D = LAT.dim; zg = torch.randn(K, 1, D, generator=gen)
    if m.kind == "1": return zg
    tau = m.tau_comp().detach(); return zg + tau * torch.randn(K, C, D, generator=gen)
def decode(m, z, u=None, chunk=int(os.environ.get('DEC_CHUNK', 500))):
    out = []
    for i in range(0, len(z), chunk):
        with torch.no_grad(): out.append(m.decode(z[i:i + chunk], u[i:i + chunk]) if m.kind == "6u" else m.decode(z[i:i + chunk]))
    return torch.cat(out)                                                  # [K,C,3,80] or [K,C,P,80]

def response_terms(lam, mult, beta):
    """lam [K,C,P,80] -> dict of chi pieces in the sample frame.  Per z: Msum, Ssum (sum over sites with multiplicity)."""
    m1, S2 = site_moments(lam); K = lam.shape[0]
    Msum = mult * m1.sum((1, 2)); Ssum = mult * S2.sum((1, 2)); Cov = Ssum - mult * torch.einsum("kcpa,kcpb->kab", m1, m1)
    w, V = torch.linalg.eigh(Ssum); nu = V[:, :, 0]; F = V[:, :, [1, 2, 0]]                                  # columns e1, e2, nu
    Cf = torch.einsum("kia,kij,kjb->kab", F, Cov, F); Mf = torch.einsum("kia,ki->ka", F, Msum)
    dec_in = 0.5 * (Cf[:, 0, 0] + Cf[:, 1, 1]); dec_nu = Cf[:, 2, 2]; lat_in = 0.5 * (Mf[:, 0] ** 2 + Mf[:, 1] ** 2); lat_nu = Mf[:, 2] ** 2
    s = beta / N_SITES; se = lambda x: float(x.std() / math.sqrt(len(x)) * s)
    return dict(dec_in=float(dec_in.mean() * s), dec_nu=float(dec_nu.mean() * s), lat_in=float(lat_in.mean() * s), lat_nu=float(lat_nu.mean() * s),
                dec_in_se=se(dec_in), dec_nu_se=se(dec_nu), lat_in_se=se(lat_in), lat_nu_se=se(lat_nu), Mbar_abs=float(Msum.norm(dim=-1).mean() / N_SITES), spin_len=float(m1.norm(dim=-1).mean()))

# ------------------------------------------------------------------ aggregated posterior
_VAL = {}
def val_data(T):
    if T not in _VAL:
        from train_k6u import oracle_u
        S = torch.tensor(load_T(T, VAL_SEEDS, root=f"{ROOT}/data_kag/seeds")); _VAL[T] = (S, torch.as_tensor(oracle_u(S.numpy())))
    return _VAL[T]
def sample_posterior(m, nrep, gen):
    """z ~ q(z|n_data) for the validation seeds (160 snapshots), nrep draws each; returns z [160*nrep,C,64] (+ u for kind 6u)."""
    S, U = val_data(m.T); F = kag_features(S, m.R); mu_c, lv_c = m.encode(F); zs = []; us = []
    for _ in range(nrep):
        e = torch.randn(mu_c.shape, generator=gen); zs.append(mu_c + e * torch.exp(0.5 * m.expand_logvar(lv_c)))
    z = torch.cat(zs)
    if m.kind == "6u":
        from train_k6u import site_vec
        u = torch.cat([site_vec(U, m.R)] * nrep); return z, u
    return z, None
def prior_u(m, K, gen):
    """u_c fields from random validation snapshots (for kind 6u with prior-sampled z; orientation of z and u are then unrelated)."""
    S, U = val_data(m.T); from train_k6u import site_vec; idx = torch.randint(0, len(U), (K,), generator=gen); return site_vec(U[idx], m.R)

# ------------------------------------------------------------------ generation check (task A, check 1)
@torch.no_grad()
def check_generation(m, z, lam, ndraw=5, gen=None, u=None):
    """sample spins n_i ~ p(n|lam_i) (quadrature-point sampling), M = sum n_i; (beta/N) <M_a M_b> in the sample frame of the realised spins (as for the MC data)."""
    mult = int(mult_of(m)); beta = 1 / m.T; K, C, P, _ = lam.shape; acc = []
    for k in range(K):
        pk = torch.softmax(lam[k].reshape(-1, lam.shape[-1]).double() @ YQ.T + LW, -1)                  # [C*P, Q]
        for _ in range(ndraw):
            idx = torch.multinomial(pk.repeat_interleave(mult, 0), 1, generator=gen).squeeze(1); X = XQ[idx]                    # all sites
            w, V = torch.linalg.eigh(X.T @ X); F = V[:, [1, 2, 0]]; Mf = X.sum(0) @ F; acc.append(torch.outer(Mf, Mf) / N_SITES * beta)
    A = torch.stack(acc).mean(0); return dict(gen_in=float(0.5 * (A[0, 0] + A[1, 1])), gen_nu=float(A[2, 2]))
