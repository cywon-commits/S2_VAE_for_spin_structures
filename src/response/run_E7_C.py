"""E7 task 1-c: latent HMC under the anisotropy field with the refitted (rotation-augmented GMM) prior on z_g.  python3 run_E7_C.py RUN nwarm nsamp nchain [D list]"""
import sys, json, os, time, warnings
warnings.filterwarnings("ignore")
from e7_prior import *
from run_E6_B import to_z, logw_obs, DS
def load_gp(name):
    d = np.load(f"{ROOT}/results/response/E7_gmm_{name}.npz")
    class G: pass
    g = G(); g.weights_ = d["w"]; g.means_ = d["mu"]; g.precisions_cholesky_ = d["L"]; g.covariances_ = None
    gp = GMMPrior.__new__(GMMPrior); gp.w = torch.tensor(d["w"], dtype=torch.float64).log(); gp.mu = torch.tensor(d["mu"], dtype=torch.float64); gp.L = torch.tensor(d["L"], dtype=torch.float64); gp.ld = torch.log(torch.diagonal(gp.L, dim1=1, dim2=2)).sum(1)
    cov = np.linalg.inv(np.einsum("kij,klj->kil", d["L"], d["L"]))                                                         # Sigma = (L L^T)^-1
    gp.cov = torch.tensor(cov, dtype=torch.float64); gp.wn = d["w"]; return gp
def draw_zg(gp, n, gen):
    k = torch.multinomial(torch.tensor(gp.wn), n, replacement=True, generator=gen); L = torch.linalg.cholesky(gp.cov[k]); return gp.mu[k] + torch.einsum("nij,nj->ni", L, torch.randn(n, 10, generator=gen, dtype=torch.float64))
def embed(x10):
    z = torch.zeros(len(x10), 1, 64, dtype=x10.dtype); z[:, 0, ACTIVE] = x10; return z
def hmc(m, gp, D, nwarm, nsamp, B, gen, nleap=12):
    C = m.R ** 3; ez = torch.tensor([[0., 0., 1.]], dtype=torch.float64).expand(B, 3)
    def U(a, b):
        a = a.detach().requires_grad_(True); b = b.detach().requires_grad_(True); z = to_z(m, embed(a).float(), b.float()); lp = gp.logpdf(a) - 0.5 * (b ** 2).sum((1, 2)) + logw_obs(m, z, D, ez)
        g = torch.autograd.grad(lp.sum(), [a, b]); return lp.detach(), g
    zg = draw_zg(gp, B, gen); eps = torch.randn(B, C, 64, generator=gen, dtype=torch.float64); step = 0.02; lp, g = U(zg, eps); accs = []; obs = []
    for it in range(nwarm + nsamp):
        pa = torch.randn_like(zg); pb = torch.randn_like(eps); a, b = zg.clone(), eps.clone(); H0 = lp - 0.5 * (pa ** 2).sum(1) - 0.5 * (pb ** 2).sum((1, 2)); ga, gb = g
        pa = pa + 0.5 * step * ga; pb = pb + 0.5 * step * gb
        for l in range(nleap):
            a = a + step * pa; b = b + step * pb; lp1, g1 = U(a, b); f = 1.0 if l < nleap - 1 else 0.5; pa = pa + f * step * g1[0]; pb = pb + f * step * g1[1]
        H1 = lp1 - 0.5 * (pa ** 2).sum(1) - 0.5 * (pb ** 2).sum((1, 2)); ok = torch.isfinite(H1) & (torch.log(torch.rand(B, generator=gen, dtype=torch.float64)) < (H1 - H0)); acc = ok.double().mean().item()
        zg = torch.where(ok[:, None], a, zg); eps = torch.where(ok[:, None, None], b, eps); lp = torch.where(ok, lp1, lp); g = (torch.where(ok[:, None], g1[0], g[0]), torch.where(ok[:, None, None], g1[1], g[1]))
        if it < nwarm: step *= math.exp(0.5 * (acc - 0.7))
        else:
            accs.append(acc)
            with torch.no_grad(): _, o = logw_obs(m, to_z(m, embed(zg).float(), eps.float()), D, ez, True); obs.append(o)
    O = torch.stack(obs); x = O[:, :, 0].mean(1); x = x - x.mean(); n = len(x); ac = [float((x[:n - k] * x[k:]).sum() / (x * x).sum().clamp_min(1e-30)) for k in range(1, min(60, n // 2))]
    ti = 1 + 2 * sum(ac[:next((i for i, v in enumerate(ac) if v < 0.05), len(ac))]); h = O[n // 2:]
    return dict(D=D, S=float(h[:, :, 0].mean()), A=float(h[:, :, 1].mean()), B=float(h[:, :, 2].mean()), acc=float(np.mean(accs)), step=float(step), tau_int=float(ti), S_chain_sd=float(h[:, :, 0].mean(0).std()), A_chain_sd=float(h[:, :, 1].mean(0).std()), B_chain_sd=float(h[:, :, 2].mean(0).std()))
if __name__ == "__main__":
    path = sys.argv[1]; nw, ns, B = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); Ds = [float(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else DS
    name = os.path.basename(path)[:-3]; m = load_nem(f"{ROOT}/{path}"); gp = load_gp(name + os.environ.get("GMM_TAG", "")); gen = torch.Generator().manual_seed(3); out = dict(run=name, T=m.T, R=m.R, hmc=[]); fn = f"{ROOT}/results/response/E7_C_{name}{os.environ.get('GMM_TAG', '')}.json"
    for D in Ds:
        t0 = time.time(); r = hmc(m, gp, D, nw, ns, B, gen); out["hmc"].append(r); json.dump(out, open(fn, "w"), indent=1); print(name, {k: round(v, 4) for k, v in r.items()}, f"{time.time()-t0:.0f}s", flush=True)
