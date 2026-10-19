"""E3 v2 D: decoder-frame oracle with refit.  For each site the frame of the decoded (posterior-mean) site distribution: z' = its mean direction, x' = cell plane normal nu (min eigenvector of the
cell-pooled decoded Q tensor) orthogonalised against z', y' = z' x x'.  Data spins are rotated into these frames and a multipole family (fitlib.Fitter) of order L is refit by ML (pooled over all sites).
python3 oracle_dec.py out.json T [nconf]   (model results/kag/kag6_T{T}_R4.pt)"""
import sys, json, numpy as np, torch
sys.path.insert(0, "..")
from fitlib import Fitter
from kmodel import kag_features
from kmodel6 import KVAE6
from eval_k import XQ, YQ, LW
from kdata import load_T, VAL_SEEDS
def run(T, nconf=40, R=4):
    ck = torch.load(f"../../results/kag/kag6_T{T:.4f}_R{R}.pt", weights_only=False); m = KVAE6(R=R, order=131); m.load_state_dict({k: v for k, v in ck["state"].items() if not k.startswith("logz.")}, strict=False); m.eval()
    S = torch.tensor(load_T(T, VAL_SEEDS)); S = S[torch.linspace(0, len(S) - 1, nconf).long()]; b = 12 // R
    with torch.no_grad():
        mu, lv = m.encode(kag_features(S, R)); lam = m.decode(mu).double()                      # [C,cells,P,80]
        p = torch.softmax(lam @ YQ.T + LW, -1); mean_dir = torch.einsum("...q,qi->...i", p, XQ); mean_dir = mean_dir / mean_dir.norm(dim=-1, keepdim=True)    # [C,cells,P,3]
        Qc = torch.einsum("...q,qi,qj->...ij", p.mean(-2), XQ, XQ); nu = torch.linalg.eigh(Qc)[1][..., :, 0]                            # [C,cells,3]
    z = mean_dir.numpy(); nu = nu.numpy()[:, :, None, :]; x = nu - (nu * z).sum(-1, keepdims=True) * z; x /= np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12; y = np.cross(z, x)
    X = S.numpy().astype(float); C = len(X); X = X.reshape(C, R, b, R, b, 3, 3).transpose(0, 1, 3, 2, 4, 5, 6).reshape(C, R * R, b * b * 3, 3)
    P = np.stack([(X * x).sum(-1), (X * y).sum(-1), (X * z).sum(-1)], -1).reshape(-1, 3); out = {}
    for L in (1, 2, 3, 4, 6, 8): out[str(L)] = Fitter(L, order=131 if L >= 8 else 59, alpha=1e-8).fit(P).nll; print(T, L, out[str(L)], flush=True)
    return dict(T=T, R=R, nconf=C, nll_by_L=out)
if __name__ == "__main__":
    r = run(float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 40); json.dump(r, open(sys.argv[1], "w"), indent=1)
