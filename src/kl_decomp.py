"""v5 1.1: decompose the cell KL of the subspace-covariance up/down model into (rotation / amplitude / biaxial) x (mean term / variance term) + other channels.
For each subspace b (projectors of mu_g, dims d_b = 2,1,2), with q(z_c) projectors replaced by those of mu_g (they are nearly aligned; the residual vs the exact KL is reported):
  mean_b = 1/2 tau_b^-2 |P_b (mu_c - mu_g)|^2
  var_b  = 1/2 d_b [ tau_b^-2 (s_b^2 + sg_b^2) - 1 - ln(s_b^2 / tau_b^2) ]            (s: cell posterior std, sg: global posterior std in subspace b)
python3 kl_decomp.py out.json run.pt ..."""
import sys, json, numpy as np, torch
from mvae2 import cell_features
from mvae7 import MVAE7
torch.set_num_threads(1)
NAMES = ["rotation", "amplitude", "biaxial"]
def load7(path):
    ck = torch.load(path, weights_only=False); a = ck["args"]; m = MVAE7(L=12, R=a["R"]); m.load_state_dict(ck["state"], strict=False); m.eval(); return m, a
@torch.no_grad()
def decompose(path):
    m, a = load7(path); R = a["R"]; S = torch.tensor(np.load(f"{a['data']}/ll_L12_T{a['T']:.3f}_{a['st']}.npy")); nv = max(len(S) // 10, 10); Sv = S[-nv:]; F = cell_features(Sv, R)
    mu_g, lv_g, la_g = m.encode3(F.mean(1, keepdim=True)); mu_c, lv_c, la_c = m.encode3(F); o = m.act
    exact = m.kl_cell(mu_g, lv_g, la_g, mu_c, lv_c, la_c).mean().item()
    Pg = m.projectors(mu_g[..., o:o + 5]); t = m.log_tau3.exp(); dims = [2., 1., 2.]; v = (mu_c - mu_g)[..., o:o + 5]
    mean_b = []; var_b = []
    for b in range(3):
        pv = torch.einsum("...ij,...j->...i", Pg[b].expand(*v.shape[:-1], 5, 5), v)
        mean_b.append((0.5 * (pv ** 2).sum(-1) / t[b] ** 2).sum(-1).mean().item())
        s2 = la_c[..., b].exp(); sg2 = la_g[..., b].exp()
        var_b.append((0.5 * dims[b] * ((s2 + sg2) / t[b] ** 2 - 1 - torch.log(s2 / t[b] ** 2))).sum(-1).mean().item())
    # other channels (isotropic, as in MVAE5)
    tau = m.tau_comp(); zm = m.zmask.clone(); zm[o:o + 5] = 0; lvg = m.expand_logvar(lv_g); lvc = m.expand_logvar(lv_c)
    other = ((torch.log(tau) - 0.5 * lvc + (lvc.exp() + (mu_c - mu_g) ** 2 + lvg.exp()) / (2 * tau ** 2) - 0.5) * zm).sum((-1, -2)).mean().item()
    tot = sum(mean_b) + sum(var_b) + other
    return dict(run=path.split("/")[-1][:-3], T=a["T"], R=R, kl_cell_exact=exact, kl_cell_decomposed=tot, residual=exact - tot, mean_terms=dict(zip(NAMES, mean_b)), var_terms=dict(zip(NAMES, var_b)),
                per_subspace={n: mean_b[i] + var_b[i] for i, n in enumerate(NAMES)}, other_channels=other, tau=t.tolist(), s_cell=torch.exp(0.5 * la_c).mean((0, 1)).tolist(), s_global=torch.exp(0.5 * la_g).mean((0, 1)).tolist())
if __name__ == "__main__":
    out = [decompose(p) for p in sys.argv[2:]]
    for r in out:
        print(r["run"], "exact %.1f decomposed %.1f (resid %.1f) | per subspace %s | other %.1f" % (r["kl_cell_exact"], r["kl_cell_decomposed"], r["residual"], {k: round(v, 1) for k, v in r["per_subspace"].items()}, r["other_channels"]), flush=True)
        print("   mean terms", {k: round(v, 1) for k, v in r["mean_terms"].items()}, "var terms", {k: round(v, 1) for k, v in r["var_terms"].items()})
    json.dump(out, open(sys.argv[1], "w"), indent=1)
