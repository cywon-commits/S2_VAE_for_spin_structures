"""E7 task 2: wave-vector resolved channel susceptibilities chi(q) of the LL nematic (MC) and of the VAEs (decoder part q-independent; latent part from cell-constant <O>).
Channels (director frame per configuration; e1 = lab-x projected on plane perp d, e2 = d x e1): Q: P2(n.d); D: 2(n.d)(n.e1), 2(n.d)(n.e2); B: (n.e1)^2-(n.e2)^2, 2(n.e1)(n.e2).
chi(q) = (beta/N) <|sum_i dO_i e^{iq r_i}|^2> (two-component channels: mean of components), q = 2 pi k/12."""
import sys, os, math, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
CH = {"Q": 1, "D": 2, "B": 2}; NCOMP = 5; L = 12; N = L ** 3
KI = np.fft.fftfreq(L, 1.0 / L).astype(int); K2 = KI[:, None, None] ** 2 + KI[None, :, None] ** 2 + KI[None, None, :] ** 2       # integer |k|^2 on the FFT grid
def frame_np(X):
    Q = 1.5 * X.T @ X / len(X) - 0.5 * np.eye(3); w, V = np.linalg.eigh(Q); d = V[:, 2]; t = np.array([1., 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1., 0])
    e1 = t - (t @ d) * d; e1 /= np.linalg.norm(e1); return d, e1, np.cross(d, e1)
def ops_np(X, d, e1, e2):
    a0, a1, a2 = X @ d, X @ e1, X @ e2; return np.stack([1.5 * a0 ** 2 - 0.5, 2 * a0 * a1, 2 * a0 * a2, a1 ** 2 - a2 ** 2, 2 * a1 * a2], -1)            # [N,5]
def mc_chiq(T):
    S = np.load(f"{ROOT}/data/ll_L12_T{T:.3f}_o.npy").astype(float); b = 1 / T; acc = np.zeros((NCOMP, L, L, L)); sq = np.zeros(NCOMP); mean = np.zeros(NCOMP); fields = []
    O = []
    for s in S:
        X = s.reshape(-1, 3); d, e1, e2 = frame_np(X); O.append(ops_np(X, d, e1, e2).reshape(L, L, L, NCOMP))
    O = np.array(O); mu = O.mean((0, 1, 2, 3)); dO = O - mu                                                                                                  # global mean removed (as the FDT variances)
    F = np.fft.fftn(dO, axes=(1, 2, 3)); chi = b / N * (np.abs(F) ** 2).mean(0).transpose(3, 0, 1, 2)                                                           # [5,L,L,L]
    site = b * (dO ** 2).mean((0, 1, 2, 3)); return dict(chi=chi, site=site, mean=mu, T=T)
def comb(chi, site=None):
    """channel arrays [3,...] (Q, D, B) from 5-component arrays"""
    c = np.stack([chi[0], 0.5 * (chi[1] + chi[2]), 0.5 * (chi[3] + chi[4])]); return c if site is None else np.array([site[0], 0.5 * (site[1] + site[2]), 0.5 * (site[3] + site[4])])
def shells(arr):
    """arr [...,L,L,L] -> (k2 values, shell means)"""
    u = np.unique(K2); return u, np.stack([arr[..., K2 == v].mean(-1) for v in u], -1)
def form_factor2(b):
    q = 2 * math.pi * KI / L; f = np.where(np.abs(np.sin(q / 2)) < 1e-12, 1.0, np.sin(q * b / 2) / (b * np.sin(q / 2) + 1e-300)); return (f[:, None, None] * f[None, :, None] * f[None, None, :]) ** 2
# ------------------------------------------------------------------ VAE
sys.path.insert(0, HERE)
from nem_response import *
@torch.no_grad()
def vae_chiq(m, z, chunk=100):
    """returns dec [5], lat [5,L,L,L] (per k, variance over z of |FT sum_i <O_i>|^2 * beta/N), site-mean operator values; all in the E6 frame rule"""
    beta = 1 / m.T; R = m.R; s = L // R; mult = s ** 3; dec = torch.zeros(NCOMP, dtype=torch.float64); ft2 = torch.zeros(NCOMP, L, L, L, dtype=torch.float64); ftm = torch.zeros(NCOMP, L, L, L, dtype=torch.complex128); n = 0
    xh = torch.tensor([1.0, 0, 0], dtype=torch.float64); yh = torch.tensor([0, 1.0, 0], dtype=torch.float64)
    for i in range(0, len(z), chunk):
        lam = m.decode_cells(z[i:i + chunk]); p = probs(lam); k = len(p)
        S2 = torch.einsum("kcq,qa,qb->kab", p, XQ, XQ) * mult; w, V = torch.linalg.eigh(S2); d = V[:, :, 2]
        e1 = xh - (d @ xh)[:, None] * d; bad = e1.norm(dim=-1) < 1e-3; e1 = torch.where(bad[:, None], yh - (d @ yh)[:, None] * d, e1); e1 = e1 / e1.norm(dim=-1, keepdim=True); e2 = torch.cross(d, e1, dim=-1)
        a0, a1, a2 = XQ @ d.T, XQ @ e1.T, XQ @ e2.T                                           # [Q,k]
        f = torch.stack([1.5 * a0 ** 2 - 0.5, 2 * a0 * a1, 2 * a0 * a2, a1 ** 2 - a2 ** 2, 2 * a1 * a2], 0).permute(0, 2, 1)       # [5,k,Q]
        mu = torch.einsum("kcq,fkq->kfc", p, f); var = torch.einsum("kcq,fkq->kfc", p, f * f) - mu ** 2                      # [k,5,C]
        dec += (mult * var.sum(2)).sum(0)
        g = mu.reshape(k, NCOMP, R, R, R).repeat_interleave(s, 2).repeat_interleave(s, 3).repeat_interleave(s, 4)             # site field [k,5,L,L,L]
        F = torch.fft.fftn(g, dim=(2, 3, 4)); ft2 += (F.abs() ** 2).sum(0); ftm += F.sum(0); n += k
    dec = dec / n * beta / N; mean = ftm / n; lat = (ft2 / n - mean.abs() ** 2) * beta / N; return dec.numpy(), lat.numpy()
