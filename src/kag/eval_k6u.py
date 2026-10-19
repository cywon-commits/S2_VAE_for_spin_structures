"""E3 v4 evaluation of cap-direction-input models.  python3 eval_k6u.py out.json T mode      mode = oracle | shuffled | nolatent
Per-config NLL/KL (posterior samples, 2 draws), decoded site distributions at the posterior mean (mean-vector length, <theta^2> about the mean direction), arrays for the regression -> cfgarr_k6u_*.npz."""
import sys, json, numpy as np, torch
from kmodel import kag_features
from kmodel6 import site_Y
from kmodel6u import KVAE6U
from train_k6u import oracle_u, site_vec, shuffle_sites
from eval_k import YQ, LW, XQ
from kdata import load_T, VAL_SEEDS
torch.set_num_threads(1)
@torch.no_grad()
def run(T, mode, R=4, nsamp=2, CH=20):
    ck = torch.load(f"../../results/kag/kag6u_T{T:.4f}_R{R}_{mode}.pt", weights_only=False); m = KVAE6U(R=R, order=131); m.load_state_dict({k: v for k, v in ck["state"].items() if not k.startswith("logz.")}, strict=False); m.eval()
    S = torch.tensor(load_T(T, VAL_SEEDS)); U = torch.as_tensor(oracle_u(S.numpy()))
    if mode == "shuffled": U = shuffle_sites(U, torch.Generator().manual_seed(123))
    u = site_vec(U, R); F = kag_features(S, R); mu_g, lv_g = m.encode(F.mean(1, keepdim=True)); mu_c, lv_c = m.encode(F); kl_g, kl_c = m.kl_terms(mu_g, lv_g, mu_c, lv_c); kl = kl_g + kl_c
    if mode == "nolatent": kl = torch.zeros(len(S))
    torch.manual_seed(0); nll = torch.zeros(len(S))
    for _ in range(nsamp):
        for i in range(0, len(S), CH): sl = slice(i, i + CH); nll[sl] += m.nll_cfg(m.decode(m.sample(mu_c[sl], lv_c[sl]), u[sl]), site_Y(S[sl], R)) / nsamp
    rho, th2 = [], []
    for i in range(0, len(S), CH):
        sl = slice(i, i + CH); lam = m.decode(mu_c[sl], u[sl]).double(); p = torch.softmax(lam @ YQ.T + LW, -1); md = torch.einsum("...q,qi->...i", p, XQ); r = md.norm(dim=-1); uu = md / r.unsqueeze(-1)
        ang = torch.acos((XQ @ uu.unsqueeze(-1)).squeeze(-1).clamp(-1, 1)); rho.append(r.reshape(-1)); th2.append((p * ang ** 2).sum(-1).reshape(-1))
    rho = torch.cat(rho); th2 = torch.cat(th2)
    np.savez(f"../../results/kag/cfgarr_k6u_T{T:.4f}_R{R}_{mode}.npz", nll=nll.numpy(), kl=kl.numpy())
    nseed = len(VAL_SEEDS); per = len(S) // nseed; sm = lambda v: v.reshape(nseed, per).mean(1)
    return dict(T=T, mode=mode, nll=(nll / 432).mean().item(), nll_se=(sm(nll) / 432).std().item() / np.sqrt(nseed), kl=kl.mean().item(), kl_se=sm(kl).std().item() / np.sqrt(nseed), mean_vector_len=rho.mean().item(), theta2=th2.mean().item())
if __name__ == "__main__":
    r = run(float(sys.argv[2]), sys.argv[3]); json.dump(r, open(sys.argv[1], "w"), indent=1); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
