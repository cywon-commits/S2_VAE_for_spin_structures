"""E6 task B: response of the nematic VAE ensembles (prior p_theta) to dH = -D sum P2(n.e), e = z: (1) exact importance reweighting of prior samples (random e per z),
(2) latent HMC in the non-centred coordinates (z_g, eps).   python3 run_E6_B.py RUN [nwarm nsamp nchain] -> results/response/E6_B_<RUN>.json"""
import sys, json, os, time
from e6_response import *
DS = [0, 0.002, -0.002, 0.005, -0.005, 0.01, -0.01, 0.02, -0.02, 0.05, -0.05]
def y2(e): return sh([2], e)                                                          # [...,5]
def to_z(m, zg, eps):
    """non-centred map (z_g, eps) -> z_c [B,C,64]; MVAE3: z_g + tau*eps; MVAE8: main channel = z_g + tau_rot P_rot eps + tau_amp P_amp eps (projectors detached)."""
    tau = m.tau_comp().detach(); z = zg + tau * eps
    if m.cls == "MVAE8":
        a = m.act; tr, ta, _ = m.log_tau3.exp().detach(); Pr, Pa, _ = m.projectors(zg[..., a:a + 5].detach()); e5 = eps[..., a:a + 5]
        z = z.clone(); z[..., a:a + 5] = zg[..., a:a + 5] + tr * torch.einsum("kij,kcj->kci", Pr[:, 0], e5) + ta * torch.einsum("kij,kcj->kci", Pa[:, 0], e5)
    return z * m.zmask
def logw_obs(m, z, D, e, obs=False):
    """log w(z;D) (sum over cells with multiplicity); with obs=True also (S, A, B) for the D-ensemble given z.  e [k,3] unit."""
    beta = 1 / m.T; mult = (12 // m.R) ** 3; lam = m.decode_cells(z).double(); d2 = (beta * D * 4 * math.pi / 5) * y2(e.double())               # [k,5]
    lh = lam.clone(); lh[..., 3:8] = lh[..., 3:8] + d2[:, None, :]; a0 = lam @ YQ.T + LWQ; a1 = lh @ YQ.T + LWQ
    lw = mult * (torch.logsumexp(a1, -1) - torch.logsumexp(a0, -1)).sum(1)
    if not obs: return lw
    with torch.no_grad():
        p = torch.softmax(a1, -1); S2 = torch.einsum("kcq,qa,qb->kab", p, XQ, XQ) * mult; w, V = torch.linalg.eigh(S2); d = V[:, :, 2]
        f = 1.5 * (d @ XQ.T) ** 2 - 0.5; S = mult * torch.einsum("kcq,kq->k", p, f) / N; A = 1.5 * (d * e.double()).sum(-1) ** 2 - 0.5
        c = torch.cross(d, e.double(), dim=-1); nc = c.norm(dim=-1, keepdim=True); ep = torch.where(nc > 1e-6, c / nc.clamp_min(1e-12), torch.tensor([1.0, 0, 0], dtype=torch.float64).expand_as(c))
        fB = (XQ @ e.double().T).T ** 2 - (XQ @ ep.T).T ** 2; B = mult * torch.einsum("kcq,kq->k", p, fB) / N
    return lw, torch.stack([S, A, B], 1)
@torch.no_grad()
def reweight(m, K=4000, ch=200, seed=5):
    g = torch.Generator().manual_seed(seed); z = sample_prior(m, K, g); e = torch.randn(K, 3, generator=g, dtype=torch.float64); e /= e.norm(dim=-1, keepdim=True); res = []
    for D in DS:
        lws = []; os_ = []
        for i in range(0, K, ch): lw, o = logw_obs(m, z[i:i + ch], D, e[i:i + ch], True); lws.append(lw); os_.append(o)
        lw = torch.cat(lws); o = torch.cat(os_); w = (lw - lw.max()).exp(); s = w.sum(); mo = (w[:, None] * o).sum(0) / s
        res.append(dict(D=D, S=float(mo[0]), A=float(mo[1]), B=float(mo[2]), ess=float(s * s / (w * w).sum())))
    return res
def hmc(m, D, nwarm, nsamp, B, gen, nleap=12):
    C = m.R ** 3; ez = torch.tensor([[0., 0., 1.]], dtype=torch.float64).expand(B, 3)
    def U(zg, eps):
        zg = zg.detach().requires_grad_(True); eps = eps.detach().requires_grad_(True); z = to_z(m, zg.float(), eps.float()); lp = -0.5 * (zg ** 2).sum((1, 2)) * 1.0 - 0.5 * (eps ** 2).sum((1, 2)) + logw_obs(m, z, D, ez)
        g = torch.autograd.grad(lp.sum(), [zg, eps]); return lp.detach(), g
    zg = torch.randn(B, 1, 64, generator=gen, dtype=torch.float64) * m.zmask.double(); eps = torch.randn(B, C, 64, generator=gen, dtype=torch.float64); step = 0.02; lp, g = U(zg, eps); accs = []; obs = []
    for it in range(nwarm + nsamp):
        pa = torch.randn_like(zg); pb = torch.randn_like(eps); a, b = zg.clone(), eps.clone(); H0 = lp - 0.5 * (pa ** 2).sum((1, 2)) - 0.5 * (pb ** 2).sum((1, 2)); ga, gb = g
        pa = pa + 0.5 * step * ga; pb = pb + 0.5 * step * gb
        for l in range(nleap):
            a = a + step * pa; b = b + step * pb; lp1, g1 = U(a, b); f = 1.0 if l < nleap - 1 else 0.5; pa = pa + f * step * g1[0]; pb = pb + f * step * g1[1]
        H1 = lp1 - 0.5 * (pa ** 2).sum((1, 2)) - 0.5 * (pb ** 2).sum((1, 2)); ok = torch.isfinite(H1) & (torch.log(torch.rand(B, generator=gen, dtype=torch.float64)) < (H1 - H0)); acc = ok.double().mean().item(); k3 = ok[:, None, None]
        zg = torch.where(k3, a, zg); eps = torch.where(k3, b, eps); lp = torch.where(ok, lp1, lp); g = (torch.where(k3, g1[0], g[0]), torch.where(k3, g1[1], g[1]))
        if it < nwarm: step *= math.exp(0.5 * (acc - 0.7))
        else:
            accs.append(acc)
            with torch.no_grad(): _, o = logw_obs(m, to_z(m, zg.float(), eps.float()), D, ez, True); obs.append(o)
    O = torch.stack(obs); x = O[:, :, 0].mean(1); x = x - x.mean(); n = len(x); ac = [float((x[:n - k] * x[k:]).sum() / (x * x).sum().clamp_min(1e-30)) for k in range(1, min(60, n // 2))]
    ti = 1 + 2 * sum(ac[:next((i for i, v in enumerate(ac) if v < 0.05), len(ac))]); h = O[n // 2:]
    return dict(D=D, S=float(h[:, :, 0].mean()), A=float(h[:, :, 1].mean()), B=float(h[:, :, 2].mean()), acc=float(np.mean(accs)), step=float(step), tau_int=float(ti), S_chain_sd=float(h[:, :, 0].mean(0).std()), A_chain_sd=float(h[:, :, 1].mean(0).std()), B_chain_sd=float(h[:, :, 2].mean(0).std()))
if __name__ == "__main__":
    path = sys.argv[1]; nw = int(sys.argv[2]) if len(sys.argv) > 2 else 150; ns = int(sys.argv[3]) if len(sys.argv) > 3 else 200; B = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    m = load_nem(f"{ROOT}/{path}"); name = os.path.basename(path)[:-3]; gen = torch.Generator().manual_seed(2); out = dict(run=name, T=m.T, R=m.R, reweight=reweight(m), hmc=[]); fn = f"{ROOT}/results/response/E6_B_{name}.json"; fn = fn if len(sys.argv) <= 5 else fn.replace(".json", "_part.json")
    print(name, "reweight", [(r["D"], round(r["S"], 3), round(r["ess"])) for r in out["reweight"]], flush=True)
    for D in ([float(x) for x in sys.argv[5].split(',')] if len(sys.argv) > 5 else DS):
        t0 = time.time(); r = hmc(m, D, nw, ns, B, gen); out["hmc"].append(r); json.dump(out, open(fn, "w"), indent=1); print(name, {k: round(v, 4) for k, v in r.items()}, f"{time.time()-t0:.0f}s", flush=True)
