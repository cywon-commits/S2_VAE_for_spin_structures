"""v3 2.2 evaluation of hierarchical-prior runs.  python3 eval_h.py out.json run1.pt run2.pt ..."""
import sys, json, numpy as np, torch
from mvae2 import compute_stats, cell_features
from mvae3 import MVAE3
from moment_vae import moments
from momlib import est_vec
torch.set_num_threads(1)
def load3(path):
    ck = torch.load(path, weights_only=False); a = ck["args"]; m = MVAE3(L=12, R=a["R"]); m.load_state_dict(ck["state"], strict=False); m.eval(); return m, a
@torch.no_grad()
def evaluate_h(path, nsamp=8):
    m, a = load3(path); S = torch.tensor(np.load(f"{a['data']}/ll_L12_T{a['T']:.3f}_{a['st']}.npy")); nv = max(len(S) // 10, 10); Sv = S[-nv:]
    R = a["R"]; nc = (12 // R) ** 3; F = cell_features(Sv, R); St = compute_stats(Sv, R); N = 1728
    mug, lvg = m.encode_g(F); md, lvd = m.encode_d(F)
    kl_g = m.kl(mug, lvg).mean().item(); kl_d = m.kl_delta(md, lvd).mean().item()
    torch.manual_seed(0); nll = 0.; each = []
    for _ in range(nsamp):
        z = m.sample(mug, lvg) + m.sample(md, lvd); lam = m.decode_cells(z)
        nll += ((-(lam * St).sum(-1) + nc * m.logz(lam)).sum(-1).mean().item() / N) / nsamp
        each.append(est_vec(*[t.numpy() for t in moments(lam)]))
    al = np.stack([e[0] for e in each]); s2 = np.stack([e[1] for e in each])
    zm = m.decode_cells(mug + md); mm = est_vec(*[t.numpy() for t in moments(zm)])
    tau = m.log_tau.exp().detach().tolist()
    # director-angle scale of tau_2: |delta| / |z_g| on the active l=2 channel (rad)
    mu_l2 = mug[..., 16:36].reshape(*mug.shape[:-1], 4, 5); nz = mu_l2.norm(dim=-1).max(-1).values.mean().item()
    return dict(run=path.split("/")[-1][:-3], T=a["T"], R=R, nll=nll, kl=kl_g + kl_d, kl_g=kl_g, kl_d=kl_d, elbo=-(nll * N + kl_g + kl_d), tau=tau,
                z_g_l2_norm=nz, tau2_angle=tau[2] / nz, a_loc_each=float(np.nanmean(al)), sigma2_each=float(np.nanmean(s2)), frac_neg=float((s2 < 0).mean()),
                a_loc_mean=float(np.nanmean(mm[0])), sigma2_mean=float(np.nanmean(mm[1])),
                delta_rms=float((md ** 2).mean().sqrt()), delta_sigma=float(torch.exp(0.5 * m.expand_logvar(lvd)).mean()))
if __name__ == "__main__":
    out = [evaluate_h(p) for p in sys.argv[2:]]
    for r in out: print(r["run"], "NLL %.4f KL %.1f (g %.1f + d %.1f) ELBO %.1f tau %s | s2 %.4f a_loc %.3f" % (r["nll"], r["kl"], r["kl_g"], r["kl_d"], r["elbo"], [round(t, 3) for t in r["tau"]], r["sigma2_each"], r["a_loc_each"]), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
