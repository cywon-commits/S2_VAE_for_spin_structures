import json, os, time, warnings
warnings.filterwarnings("ignore")
from e7_prior import *
from e6_response import response_ch
runs = [(f"results/v5a/v5a_T1.05o_R{R}_b1.pt", 1.05, R, "B") for R in (2, 4, 3)] + [(f"results/v3h/v3h_T1.05o_R{R}_b1.pt", 1.05, R, "A") for R in (2, 4, 3)] + [(f"results/v3h/v3h_T0.95o_R{R}_b1.pt", 0.95, R, "A") for R in (2, 4, 3)]
out = {}
for p, T, R, mod in runs:
    t0 = time.time(); m = load_nem(f"{ROOT}/{p}"); g = torch.Generator().manual_seed(0); Sv = val_spins(T); name = os.path.basename(p)[:-3]
    parts = [posterior_parts(m, Sv, g) for _ in range(8)]; zgf = torch.cat([a for a, _ in parts]); zc = torch.cat([b for _, b in parts]); zg = zgf[:, 0, ACTIVE]; ds = data_S(T)
    rec = dict(run=name, model=mod, T=T, R=R)
    rec["zg_var"] = zg.var(0).tolist(); rec["zg_mean"] = zg.mean(0).tolist()
    for c, off in (("ch21", 21), ("ch31", 31)):
        n = (zgf[:, 0, off:off + 5] ** 2).sum(-1); rec[c] = dict(E_n2=float(n.mean()), norm_mean=float(n.sqrt().mean()), norm_std=float(n.sqrt().std()), n2_prior=5.0)
    ep = eps_of(m, zgf, zc); rec["eps_var"] = ep[..., ACTIVE].reshape(-1, 10).var(0).tolist()
    Sp = S_of(m, zc); gz = torch.randn(len(zgf), 1, 64, generator=g) * m.zmask
    S_i = S_of(m, cond_cells(m, zgf, g)); S_ii = S_of(m, (gz + (zc - zgf)) * m.zmask); S_old = S_of(m, sample_prior(m, 2000, g))
    st = lambda x: dict(mean=float(x.mean()), std=float(x.std())); rec["S"] = dict(data=st(torch.tensor(ds)), post=st(Sp), i_zg_post_cells_prior=st(S_i), ii_zg_prior_cells_post=st(S_ii), old_prior=st(S_old))
    x = rotate_l2(zg, g, 16).numpy(); gm, k, bics = fit_gmm(x, (50, 100, 200), reg=1e-4); gp = GMMPrior(gm); rec["gmm"] = dict(k=k, bic=bics)
    np.savez(f"{ROOT}/results/response/E7_gmm_{name}.npz", w=gm.weights_, mu=gm.means_, L=gm.precisions_cholesky_)
    zn, zgn = sample_prior_new(m, 2000, g, gp); Sn = S_of(m, zn); rec["S"]["new_prior"] = st(Sn); rec["S_new_samples"] = Sn[:2000].tolist(); rec["S_data_samples"] = ds.tolist(); rec["S_old_samples"] = S_old.tolist()
    rec["resp_new"] = response_ch(m, zn); rec["resp_post"] = response_ch(m, zc); rec["resp_old"] = response_ch(m, sample_prior(m, 1000, g))
    out[name] = rec; print(name, {k: (round(v["mean"], 3), round(v["std"], 3)) for k, v in rec["S"].items()}, "gmm k", k, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(f"{ROOT}/results/response/E7_1ab.json", "w"), indent=1)
