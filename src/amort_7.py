"""v5 1.2: amortisation gap of the subspace-covariance up/down model.  Decoder fixed; per validation config, optimise q(z_g), q(z_c) directly
(mu, scalar log-variances and subspace log-variances), record the reconstruction term and the KL separately.  python3 amort_7.py out.json run.pt ..."""
import sys, json, numpy as np, torch
from mvae2 import compute_stats, cell_features
from kl_decomp import load7
torch.set_num_threads(1)
def terms(m, St, nc, q, nsamp, seed):
    mu_g, lv_g, la_g, mu_c, lv_c, la_c = q; torch.manual_seed(seed); rec = 0.
    for _ in range(nsamp):
        lam = m.decode_cells(m.sample3(mu_c, lv_c, la_c)); rec = rec + (-(lam * St).sum(-1) + nc * m.logz(lam)).sum(-1) / nsamp
    kl = m.kl3(mu_g, lv_g, la_g) + m.kl_cell(mu_g, lv_g, la_g, mu_c, lv_c, la_c)
    return rec, kl
def run(path, steps=800, ncfg=20):
    m, a = load7(path); R = a["R"]; nc = (12 // R) ** 3
    S = torch.tensor(np.load(f"{a['data']}/ll_L12_T{a['T']:.3f}_{a['st']}.npy"))[-ncfg:]; F = cell_features(S, R); St = compute_stats(S, R)
    with torch.no_grad(): q0 = [*m.encode3(F.mean(1, keepdim=True)), *m.encode3(F)]
    for p in m.parameters(): p.requires_grad_(False)
    q = [t.clone().requires_grad_(True) for t in q0]; opt = torch.optim.Adam(q, lr=5e-3)
    for s in range(steps):
        qq = [q[0], q[1].clamp(-12, 4), q[2].clamp(-12, 4), q[3], q[4].clamp(-12, 4), q[5].clamp(-12, 4)]
        torch.manual_seed(s); rec, kl = terms(m, St, nc, qq, 2, s); loss = (rec + kl).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        r0, k0 = terms(m, St, nc, q0, 32, 123); qq = [q[0].detach(), q[1].detach().clamp(-12, 4), q[2].detach().clamp(-12, 4), q[3].detach(), q[4].detach().clamp(-12, 4), q[5].detach().clamp(-12, 4)]
        r1, k1 = terms(m, St, nc, qq, 32, 123)
    return dict(run=path.split("/")[-1][:-3], T=a["T"], R=R, rec_encoder=r0.mean().item(), kl_encoder=k0.mean().item(), rec_direct=r1.mean().item(), kl_direct=k1.mean().item(),
                elbo_gain=((r0 + k0) - (r1 + k1)).mean().item(), rec_change=(r1 - r0).mean().item(), kl_change=(k1 - k0).mean().item())
if __name__ == "__main__":
    out = [run(p) for p in sys.argv[2:]]
    for r in out: print(r["run"], "ELBO gain %.1f nats/config | reconstruction %+.1f | KL %+.1f (%.1f -> %.1f)" % (r["elbo_gain"], r["rec_change"], r["kl_change"], r["kl_encoder"], r["kl_direct"]), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
