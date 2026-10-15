"""v4 task B evaluation: R=1 magnitude/direction-separated decoder.  python3 eval_6.py out.json run.pt ..."""
import sys, json, numpy as np, torch
from mvae2 import compute_stats, cell_features
from mvae6 import MVAE6
torch.set_num_threads(1)
@torch.no_grad()
def evaluate_6(path, nsamp=8):
    ck = torch.load(path, weights_only=False); a = ck["args"]; m = MVAE6(L=12, R=1); m.load_state_dict(ck["state"], strict=False); m.eval()
    S = torch.tensor(np.load(f"{a['data']}/ll_L12_T{a['T']:.3f}_{a['st']}.npy")); nv = max(len(S) // 10, 10); Sv = S[-nv:]; F = cell_features(Sv, 1); St = compute_stats(Sv, 1); N = 1728
    mu, lv = m.encode(feat=F); kl = m.kl(mu, lv).mean().item(); torch.manual_seed(0); nll = 0.
    for _ in range(nsamp):
        lam = m.decode_cells(m.sample(mu, lv)); nll += ((-(lam * St).sum(-1) + 1728 * m.logz(lam)).sum(-1).mean().item() / N) / nsamp
    lvx = m.expand_logvar(lv); mean_term = (0.5 * (mu ** 2 * m.zmask).sum(-1)).mean().item(); var_term = (0.5 * (lvx.exp() - 1 - lvx) * m.zmask).sum(-1).mean().item()
    a0 = m.act; mn = mu[..., a0:a0 + 5].norm(dim=-1)
    return dict(run=path.split("/")[-1][:-3], T=a["T"], kl=kl, nll=nll, elbo=-(nll * N + kl), kl_mean_term=mean_term, kl_var_term=var_term, mu_norm_main=mn.mean().item(),
                mu_norm_main_std=mn.std().item(), sigma_main=torch.exp(0.5 * lv[..., (a0 - 16) // 5 + 8]).mean().item(), main_channel_offset=a0)
if __name__ == "__main__":
    out = [evaluate_6(p) for p in sys.argv[2:]]
    for r in out: print(r["run"], "NLL %.4f KL %.1f (mean %.1f + var %.1f) ELBO %.1f |mu_main| %.2f +- %.2f sigma %.4f" % (r["nll"], r["kl"], r["kl_mean_term"], r["kl_var_term"], r["elbo"], r["mu_norm_main"], r["mu_norm_main_std"], r["sigma_main"]), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
