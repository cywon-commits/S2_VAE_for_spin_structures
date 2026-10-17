"""v5 evaluation of ablation (MVAE8, 'v5a') and combined (MVAE9, 'v5b') runs via their own forward pass.  python3 eval_8.py out.json run.pt ..."""
import sys, json, numpy as np, torch
from mvae2 import compute_stats, cell_features
from mvae8 import MVAE8
from mvae9 import MVAE9
torch.set_num_threads(1)
@torch.no_grad()
def evaluate_8(path, ndraw=8):
    ck = torch.load(path, weights_only=False); a = ck["args"]; R = a["R"]; cls = MVAE8 if a.get("abl") else MVAE9
    m = cls(L=12, R=R); m.load_state_dict(ck["state"], strict=False); m.eval(); nc = (12 // R) ** 3
    S = torch.tensor(np.load(f"{a['data']}/ll_L12_T{a['T']:.3f}_{a['st']}.npy")); nv = max(len(S) // 10, 10); Sv = S[-nv:]; F = cell_features(Sv, R); St = compute_stats(Sv, R)
    torch.manual_seed(0); nll = kl_g = kl_c = 0.
    for _ in range(ndraw):
        o = m.forward_stats(F, St, nc, sample=True); nll += o["nll"].item() / ndraw; kl_g += o["kl_g"].item() / ndraw; kl_c += o["kl_c"].item() / ndraw
    t3 = m.log_tau3.exp().tolist(); act = m.act
    mu_g, lv_g, la_g = m.encode3(F.mean(1, keepdim=True)); nz = mu_g[..., act:act + 5].norm(dim=-1).mean().item()
    return dict(run=path.split("/")[-1][:-3], T=a["T"], R=R, kind="ablation" if a.get("abl") else "separated+subspace", nll=nll, kl=kl_g + kl_c, kl_g=kl_g, kl_c=kl_c, elbo=-(nll * 1728 + kl_g + kl_c),
                tau_rot_amp_bi=t3, z_g_norm=nz, tau_rot_angle=t3[0] / nz)
if __name__ == "__main__":
    out = [evaluate_8(p) for p in sys.argv[2:]]
    for r in out: print(r["run"], r["kind"], "NLL %.4f KL %.1f (g %.1f + c %.1f) ELBO %.1f tau %s" % (r["nll"], r["kl"], r["kl_g"], r["kl_c"], r["elbo"], [round(x, 3) for x in r["tau_rot_amp_bi"]]), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
