"""Helpers that decode kagome VAE models for example plots (needs the local checkpoints results/kag/*.pt, which are not in git: regenerate with src/kag/train_k6.py / train_k6u.py)."""
import sys, numpy as np, torch
from style import ROOT
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/src/kag"); sys.path.insert(0, f"{ROOT}/src/layered"); sys.path.insert(0, f"{ROOT}/src/kagome")
from kmodel import kag_features, LMAX
from kmodel6 import KVAE6
from kmodel6u import KVAE6U
from train_k6u import oracle_u, site_vec
from oracle_k import to_frame
from mvae import sh
from kdata import load_T, VAL_SEEDS
def load_models(T=0.0005):
    out = {}
    for name, cls, path in (("free", KVAE6, f"results/kag/kag6_T{T:.4f}_R4.pt"), ("colour", KVAE6U, f"results/kag/kag6u_T{T:.4f}_R4_oracle.pt")):
        ck = torch.load(f"{ROOT}/{path}", weights_only=False); m = cls(R=4, order=59); m.load_state_dict({k: v for k, v in ck["state"].items() if not k.startswith("logz.")}, strict=False); m.eval(); out[name] = m
    return out
@torch.no_grad()
def decode_example(models, T=0.0005, cfg=0, cell=5, pos=4):
    """returns (lam_free, lam_colour, spins of the cell [27,3], colour vectors u of the cell [27,3], global frame [3,3] columns)."""
    S = torch.tensor(load_T(T, VAL_SEEDS, root=f"{ROOT}/data_kag/seeds"))[cfg:cfg + 1]; F = kag_features(S, 4); U = torch.as_tensor(oracle_u(S.numpy())); u = site_vec(U, 4)
    mu, lv = models["free"].encode(F); lam0 = models["free"].decode(mu)[0, cell, pos].numpy()
    mu, lv = models["colour"].encode(F); lam1 = models["colour"].decode(mu, u)[0, cell, pos].numpy()
    X = S.numpy().astype(float); b = 3; Xc = X.reshape(1, 4, b, 4, b, 3, 3).transpose(0, 1, 3, 2, 4, 5, 6).reshape(16, 27, 3)[cell]; Uc = u[0, cell].numpy()
    Fr = to_frame(X)                                                                       # data rotated into global frame; need the frame itself
    Q = np.einsum('ni,nj->ij', X.reshape(-1, 3), X.reshape(-1, 3)); w, V = np.linalg.eigh(Q); nu = V[:, 0]; e1 = V[:, 2]; e2 = np.cross(nu, e1)
    phi = np.arctan2(X.reshape(-1, 3) @ e2, X.reshape(-1, 3) @ e1); ph0 = np.angle(np.mean(np.exp(3j * phi))) / 3; xp = np.cos(ph0) * e1 + np.sin(ph0) * e2; yp = np.cross(nu, xp); frame = np.stack([xp, yp, nu], -1)
    return lam0, lam1, Xc, Uc, frame
def logp_from_lam(lam):
    lam = torch.tensor(lam, dtype=torch.float64)
    def f(V):
        Y = sh(range(1, LMAX + 1), torch.tensor(V.reshape(-1, 3), dtype=torch.float64)).numpy(); return (Y @ lam.numpy()).reshape(V.shape[:-1])
    return f
