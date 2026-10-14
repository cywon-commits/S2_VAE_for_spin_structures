"""v3 2.3 evaluation: equivariant posterior covariance (R=1).  python3 eval_s.py out.json run.pt ..."""
import sys, json, numpy as np, torch
from mvae2 import compute_stats, cell_features
from mvae4 import MVAE4
torch.set_num_threads(1)
@torch.no_grad()
def evaluate_s(path, nsamp=8):
    ck = torch.load(path, weights_only=False); a = ck["args"]; m = MVAE4(L=12, R=1); m.load_state_dict(ck["state"], strict=False); m.eval()
    S = torch.tensor(np.load(f"{a['data']}/ll_L12_T{a['T']:.3f}_{a['st']}.npy")); nv = max(len(S) // 10, 10); Sv = S[-nv:]
    F = cell_features(Sv, 1); St = compute_stats(Sv, 1); N = 1728
    mu, lv, la = m.encode3(F); kl = m.kl3(mu, lv, la).mean().item(); torch.manual_seed(0); nll = 0.
    for _ in range(nsamp):
        lam = m.decode_cells(m.sample3(mu, lv, la)); nll += ((-(lam * St).sum(-1) + 1728 * m.logz(lam)).sum(-1).mean().item() / N) / nsamp
    mn = mu[..., m.act:m.act + 5].norm(dim=-1).mean().item(); sd = torch.exp(0.5 * la).mean((0, 1))
    return dict(run=path.split("/")[-1][:-3], T=a["T"], kl=kl, nll=nll, elbo=-(nll * N + kl), main_channel_offset=m.act, mu_norm=mn,
                sigma_rot=sd[0].item(), sigma_amp=sd[1].item(), sigma_bi=sd[2].item(), sigma_rot_angle=sd[0].item() / mn)
if __name__ == "__main__":
    out = [evaluate_s(p) for p in sys.argv[2:]]
    for r in out: print(r["run"], "KL %.1f NLL %.4f ELBO %.1f | sigma rot %.4f amp %.4f bi %.4f (rot angle %.4f rad)" % (r["kl"], r["nll"], r["elbo"], r["sigma_rot"], r["sigma_amp"], r["sigma_bi"], r["sigma_rot_angle"]), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
