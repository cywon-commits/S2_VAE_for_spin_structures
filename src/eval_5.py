"""v4 task A evaluation (MVAE5 up-posterior / down-prior).  python3 eval_5.py out.json run.pt ..."""
import sys, json, numpy as np, torch
from mvae2 import compute_stats, cell_features
from mvae5 import MVAE5
from moment_vae import moments
from momlib import est_vec
torch.set_num_threads(1)
def load5(path):
    ck = torch.load(path, weights_only=False); a = ck["args"]; m = MVAE5(L=12, R=a["R"]); m.load_state_dict(ck["state"], strict=False); m.eval(); return m, a
@torch.no_grad()
def evaluate_5(path, nsamp=8):
    m, a = load5(path); S = torch.tensor(np.load(f"{a['data']}/ll_L12_T{a['T']:.3f}_{a['st']}.npy")); nv = max(len(S) // 10, 10); Sv = S[-nv:]
    R = a["R"]; nc = (12 // R) ** 3; F = cell_features(Sv, R); St = compute_stats(Sv, R); N = 1728
    mu_g, lv_g = m.encode_g(F); mu_c, lv_c = m.encode_c(F); kl_g, kl_c = m.kl_terms(mu_g, lv_g, mu_c, lv_c); kl_g = kl_g.mean().item(); kl_c = kl_c.mean().item()
    torch.manual_seed(0); nll = 0.; each = []
    for _ in range(nsamp):
        lam = m.decode_cells(m.sample(mu_c, lv_c)); nll += ((-(lam * St).sum(-1) + nc * m.logz(lam)).sum(-1).mean().item() / N) / nsamp
        each.append(est_vec(*[t.numpy() for t in moments(lam)]))
    al = np.stack([e[0] for e in each]); s2 = np.stack([e[1] for e in each]); tau = m.log_tau.exp().detach().tolist()
    mu_l2 = mu_g[..., 16:36].reshape(*mu_g.shape[:-1], 4, 5); nz = mu_l2.norm(dim=-1).max(-1).values.mean().item()
    dev = (mu_c - mu_g)[..., 16:36].reshape(*mu_c.shape[:-1], 4, 5).norm(dim=-1).max(-1).values.mean().item()   # |mu_c - mu_g| on the main l=2 channel
    return dict(run=path.split("/")[-1][:-3], T=a["T"], R=R, nll=nll, kl=kl_g + kl_c, kl_g=kl_g, kl_c=kl_c, elbo=-(nll * N + kl_g + kl_c), tau=tau, z_g_l2_norm=nz,
                tau2_angle=tau[2] / nz, dev_norm=dev, dev_angle=dev / nz, a_loc_each=float(np.nanmean(al)), sigma2_each=float(np.nanmean(s2)), frac_neg=float((s2 < 0).mean()),
                s_c=float(torch.exp(0.5 * m.expand_logvar(lv_c)).mean()))
if __name__ == "__main__":
    out = [evaluate_5(p) for p in sys.argv[2:]]
    for r in out: print(r["run"], "NLL %.4f KL %.1f (g %.1f + c %.1f) ELBO %.1f tau2 %.3f (%.3f rad) | s2 %.4f a_loc %.3f" % (r["nll"], r["kl"], r["kl_g"], r["kl_c"], r["elbo"], r["tau"][2], r["tau2_angle"], r["sigma2_each"], r["a_loc_each"]), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
