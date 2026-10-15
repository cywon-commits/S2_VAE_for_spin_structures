"""v4 extra: evaluation of the subspace-covariance up/down model (MVAE7).  python3 eval_7.py out.json run.pt ..."""
import sys, json, numpy as np, torch
from mvae2 import compute_stats, cell_features
from mvae7 import MVAE7
torch.set_num_threads(1)
@torch.no_grad()
def evaluate_7(path, nsamp=8):
    ck = torch.load(path, weights_only=False); a = ck["args"]; R = a["R"]; m = MVAE7(L=12, R=R); m.load_state_dict(ck["state"], strict=False); m.eval()
    S = torch.tensor(np.load(f"{a['data']}/ll_L12_T{a['T']:.3f}_{a['st']}.npy")); nv = max(len(S) // 10, 10); Sv = S[-nv:]; nc = (12 // R) ** 3; F = cell_features(Sv, R); St = compute_stats(Sv, R); N = 1728
    mu_g, lv_g, la_g = m.encode3(F.mean(1, keepdim=True)); mu_c, lv_c, la_c = m.encode3(F)
    kl_g = m.kl3(mu_g, lv_g, la_g).mean().item(); kl_c = m.kl_cell(mu_g, lv_g, la_g, mu_c, lv_c, la_c).mean().item(); torch.manual_seed(0); nll = 0.
    for _ in range(nsamp):
        lam = m.decode_cells(m.sample3(mu_c, lv_c, la_c)); nll += ((-(lam * St).sum(-1) + nc * m.logz(lam)).sum(-1).mean().item() / N) / nsamp
    a0 = m.act; nz = mu_g[..., a0:a0 + 5].norm(dim=-1).mean().item(); t3 = m.log_tau3.exp().tolist(); sc = torch.exp(0.5 * la_c).mean((0, 1)).tolist()
    return dict(run=path.split("/")[-1][:-3], T=a["T"], R=R, nll=nll, kl=kl_g + kl_c, kl_g=kl_g, kl_c=kl_c, elbo=-(nll * N + kl_g + kl_c), tau_iso=m.log_tau.exp().tolist(), tau_rot_amp_bi=t3,
                z_g_norm=nz, tau_rot_angle=t3[0] / nz, s_cell_rot_amp_bi=sc, s_cell_rot_angle=sc[0] / nz)
if __name__ == "__main__":
    out = [evaluate_7(p) for p in sys.argv[2:]]
    for r in out: print(r["run"], "NLL %.4f KL %.1f (g %.1f + c %.1f) ELBO %.1f tau rot/amp/bi %s (rot %.3f rad) s_cell %s" % (r["nll"], r["kl"], r["kl_g"], r["kl_c"], r["elbo"], [round(x, 3) for x in r["tau_rot_amp_bi"]], r["tau_rot_angle"], [round(x, 3) for x in r["s_cell_rot_amp_bi"]]), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
