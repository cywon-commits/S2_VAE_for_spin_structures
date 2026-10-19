"""E3 v3 evaluation of colour-conditioned site-resolved models.  python3 eval_k6c.py out.json T R color
Reports NLL/KL with the (oracle or shuffled) colours as conditioning input, paired gains against (a) the v1 R=1 baseline and (b) the v2 colour-free model of the same R, mean-vector length of decoded site
distributions, and angular spread (mean theta^2 about the mean direction) of decoded site distributions vs data colour classes (spins of one colour inside one cell).  Per-config arrays go to cfgarr_k6c_*.npz."""
import sys, json, numpy as np, torch
from kmodel import kag_features
from kmodel6 import site_Y
from kmodel6c import KVAE6C, oracle_colors, site_col
from eval_k import YQ, LW, XQ
from eval_ku import per_config as per_config_v1
from eval_k6 import per_config6
from kdata import load_T, VAL_SEEDS
torch.set_num_threads(1)
def shuffle(C, g):
    flat = C.reshape(len(C), -1).clone()
    for i in range(len(flat)): flat[i] = flat[i][torch.randperm(flat.shape[1], generator=g)]
    return flat.reshape(C.shape)
@torch.no_grad()
def run(T, R, color, nsamp=2, CH=20):
    ck = torch.load(f"../../results/kag/kag6c_T{T:.4f}_R{R}_{color}.pt", weights_only=False); m = KVAE6C(R=R, order=131); m.load_state_dict({k: v for k, v in ck["state"].items() if not k.startswith("logz.")}, strict=False); m.eval()
    S = torch.tensor(load_T(T, VAL_SEEDS)); C = torch.as_tensor(oracle_colors(S.numpy()))
    if color == "shuffled": C = shuffle(C, torch.Generator().manual_seed(123))
    col = site_col(C, R); F = kag_features(S, R); mu_g, lv_g = m.encode(F.mean(1, keepdim=True)); mu_c, lv_c = m.encode(F); kl_g, kl_c = m.kl_terms(mu_g, lv_g, mu_c, lv_c); kl = kl_g + kl_c
    torch.manual_seed(0); nll = torch.zeros(len(S))
    for _ in range(nsamp):
        for i in range(0, len(S), CH): sl = slice(i, i + CH); nll[sl] += m.nll_cfg(m.decode(m.sample(mu_c[sl], lv_c[sl]), col[sl]), site_Y(S[sl], R)) / nsamp
    # decoded site distributions at the posterior mean: mean-vector length and angular spread about the mean direction
    rho, th2 = [], []
    for i in range(0, len(S), CH):
        sl = slice(i, i + CH); lam = m.decode(mu_c[sl], col[sl]).double(); p = torch.softmax(lam @ YQ.T + LW, -1); md = torch.einsum("...q,qi->...i", p, XQ); r = md.norm(dim=-1); u = md / r.unsqueeze(-1)
        ang = torch.acos((XQ @ u.unsqueeze(-1)).squeeze(-1).clamp(-1, 1)); rho.append(r.reshape(-1)); th2.append((p * ang ** 2).sum(-1).reshape(-1))
    rho = torch.cat(rho); th2 = torch.cat(th2)
    # data colour classes: spins of one colour within one cell, angle to their own mean direction
    b = 12 // R; X = S.double().reshape(len(S), R, b, R, b, 3, 3).permute(0, 1, 3, 2, 4, 5, 6).reshape(len(S), R * R, b * b * 3, 3); Cc = C.reshape(len(S), R, b, R, b, 3).permute(0, 1, 3, 2, 4, 5).reshape(len(S), R * R, b * b * 3)
    dr, dt = [], []
    for k in range(3):
        msk = (Cc == k).double().unsqueeze(-1); n = msk.sum(2); ok = n[..., 0] >= 3; mv = (X * msk).sum(2) / n.clamp(min=1); r = mv.norm(dim=-1); u = mv / r.unsqueeze(-1).clamp(min=1e-9)
        ang = torch.acos((X * u.unsqueeze(2)).sum(-1).clamp(-1, 1)); t2 = (ang ** 2 * msk[..., 0]).sum(2) / n[..., 0].clamp(min=1); dr.append(r[ok]); dt.append(t2[ok])
    base = per_config_v1(f"../../results/kag/kag_T{T:.4f}_R1_p.pt"); ref = per_config6(f"../../results/kag/kag6_T{T:.4f}_R{R}.pt")
    nseed = len(VAL_SEEDS); per = len(S) // nseed; seedm = lambda v: v.reshape(nseed, per).mean(1); se = lambda v: v.std().item() / np.sqrt(nseed)
    eb = -(base["nll_cfg"] + base["kl_g"] + base["kl_c"]); er = -(nll + kl); ei = -(ref["nll_cfg"] + ref["kl_g"] + ref["kl_c"]); g_b = seedm(er - eb); g_i = seedm(er - ei); nl_b = seedm(base["nll_cfg"] - nll) / 432; nl_i = seedm(ref["nll_cfg"] - nll) / 432
    np.savez(f"../../results/kag/cfgarr_k6c_T{T:.4f}_R{R}_{color}.npz", nll=nll.numpy(), nll_base=base["nll_cfg"].numpy(), nll_i=ref["nll_cfg"].numpy(), kl=kl.numpy(), kl_i=(ref["kl_g"] + ref["kl_c"]).numpy())
    return dict(T=T, R=R, color=color, nll=(nll / 432).mean().item(), kl=kl.mean().item(), net_gain_vs_R1=g_b.mean().item(), net_gain_vs_R1_se=se(g_b), nll_gain_vs_R1=nl_b.mean().item(), nll_gain_vs_R1_se=se(nl_b),
                net_gain_vs_colourfree=g_i.mean().item(), net_gain_vs_colourfree_se=se(g_i), nll_gain_vs_colourfree=nl_i.mean().item(), nll_gain_vs_colourfree_se=se(nl_i),
                decoded_mean_vector_len=rho.mean().item(), decoded_theta2=th2.mean().item(), data_class_mean_vector_len=torch.cat(dr).mean().item(), data_class_theta2=torch.cat(dt).mean().item(), colourfree_nll=(ref["nll_cfg"] / 432).mean().item(), colourfree_kl=(ref["kl_g"] + ref["kl_c"]).mean().item())
if __name__ == "__main__":
    r = run(float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]); json.dump(r, open(sys.argv[1], "w"), indent=1); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
