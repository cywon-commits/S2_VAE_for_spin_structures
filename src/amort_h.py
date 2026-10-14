"""Is the encoder the bottleneck? fix the decoder, optimise q(z_g), q(delta_c) per validation config directly, compare ELBO.  python3 amort_h.py out.json run.pt ..."""
import sys, json, numpy as np, torch
from eval_h import load3
from mvae2 import compute_stats, cell_features
torch.set_num_threads(1)
def elbo(m, St, nc, mug, lvg, md, lvd, nsamp, seed):
    g = torch.Generator().manual_seed(seed); tot = 0.
    for _ in range(nsamp):
        zg = mug + torch.randn(mug.shape, generator=g) * torch.exp(0.5 * m.expand_logvar(lvg)); dl = md + torch.randn(md.shape, generator=g) * torch.exp(0.5 * m.expand_logvar(lvd))
        lam = m.decode_cells(zg + dl); tot = tot + (-(lam * St).sum(-1) + nc * m.logz(lam)).sum(-1) / nsamp
    return -tot - m.kl(mug, lvg) - m.kl_delta(md, lvd)
def run(path, steps=800, ncfg=20):
    m, a = load3(path); S = torch.tensor(np.load(f"{a['data']}/ll_L12_T{a['T']:.3f}_{a['st']}.npy")); Sv = S[-ncfg:]; R = a["R"]; nc = (12 // R) ** 3
    F = cell_features(Sv, R); St = compute_stats(Sv, R)
    with torch.no_grad(): mug0, lvg0 = m.encode_g(F); md0, lvd0 = m.encode_d(F)
    for p in m.parameters(): p.requires_grad_(False)
    v = [t.clone().requires_grad_(True) for t in (mug0, lvg0, md0, lvd0)]; opt = torch.optim.Adam(v, lr=5e-3)
    for s in range(steps):
        l = -elbo(m, St, nc, v[0], v[1].clamp(-12, 4), v[2], v[3].clamp(-12, 4), 4, s).mean(); opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        e0 = elbo(m, St, nc, mug0, lvg0, md0, lvd0, 64, 123); e1 = elbo(m, St, nc, v[0].detach(), v[1].detach().clamp(-12, 4), v[2].detach(), v[3].detach().clamp(-12, 4), 64, 123)
        kl1 = (m.kl(v[0].detach(), v[1].detach().clamp(-12, 4)) + m.kl_delta(v[2].detach(), v[3].detach().clamp(-12, 4))).mean().item()
        kl0 = (m.kl(mug0, lvg0) + m.kl_delta(md0, lvd0)).mean().item()
    return dict(run=path.split("/")[-1][:-3], T=a["T"], R=R, gain_mean=(e1 - e0).mean().item(), gain_max=(e1 - e0).max().item(), kl_encoder=kl0, kl_direct=kl1)
if __name__ == "__main__":
    out = [run(p) for p in sys.argv[2:]]
    for r in out: print(r["run"], "ELBO gain from direct posterior optimisation: mean %.1f max %.1f nats/config | KL %.1f -> %.1f" % (r["gain_mean"], r["gain_max"], r["kl_encoder"], r["kl_direct"]), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
