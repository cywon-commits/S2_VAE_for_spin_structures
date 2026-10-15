"""v4 task B control (plain decoder re-initialised, no separation): R=1 KL decomposition.  python3 eval_c.py out.json run.pt ..."""
import sys, json, numpy as np, torch
from mvae2 import MVAE2, compute_stats, cell_features
torch.set_num_threads(1)
@torch.no_grad()
def evaluate_c(path, nsamp=8):
    ck = torch.load(path, weights_only=False); a = ck["args"]; m = MVAE2(L=12, R=1); m.load_state_dict(ck["state"], strict=False); m.eval()
    S = torch.tensor(np.load(f"{a['data']}/ll_L12_T{a['T']:.3f}_{a['st']}.npy")); nv = max(len(S) // 10, 10); Sv = S[-nv:]; F = cell_features(Sv, 1); St = compute_stats(Sv, 1); N = 1728
    mu, lv = m.encode(feat=F); kl = m.kl(mu, lv).mean().item(); torch.manual_seed(0); nll = 0.
    for _ in range(nsamp):
        lam = m.decode_cells(m.sample(mu, lv)); nll += ((-(lam * St).sum(-1) + 1728 * m.logz(lam)).sum(-1).mean().item() / N) / nsamp
    lvx = m.expand_logvar(lv); mean_term = (0.5 * (mu ** 2 * m.zmask).sum(-1)).mean().item(); var_term = (0.5 * (lvx.exp() - 1 - lvx) * m.zmask).sum(-1).mean().item()
    offs = [16 + 5 * c for c in range(4) if m.zmask[16 + 5 * c:21 + 5 * c].sum() > 0]
    mnl = [mu[..., o:o + 5].norm(dim=-1).mean().item() for o in offs]; sgl = [torch.exp(0.5 * lv[..., (o - 16) // 5 + 8]).mean().item() for o in offs]
    return dict(run=path.split("/")[-1][:-3], T=a["T"], kl=kl, nll=nll, elbo=-(nll * N + kl), kl_mean_term=mean_term, kl_var_term=var_term, channel_offsets=offs, mu_norm=mnl, sigma=sgl)
if __name__ == "__main__":
    out = [evaluate_c(p) for p in sys.argv[2:]]
    for r in out: print(r["run"], "NLL %.4f KL %.1f (mean %.1f + var %.1f) ELBO %.1f |mu| %s sigma %s" % (r["nll"], r["kl"], r["kl_mean_term"], r["kl_var_term"], r["elbo"], [round(x, 2) for x in r["mu_norm"]], [round(x, 3) for x in r["sigma"]]), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
