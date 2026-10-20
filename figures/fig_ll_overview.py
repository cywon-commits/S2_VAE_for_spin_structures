"""Figures 1, 2, 3, 4, 5 (concept, symmetry rules, method checks, nematic overview, symmetry verdict)."""
import sys, json, numpy as np, torch
from style import *
from tables import md_table, num, nums
sys.path.insert(0, f"{ROOT}/src")
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
MP = {}
def fig01():
    fig = plt.figure(figsize=(W2, 3.9)); gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.12, wspace=0.05)
    # (a) schematic
    a = fig.add_subplot(gs[0, :]); a.set_xlim(0, 100); a.set_ylim(0, 24); a.axis("off")
    def box(x, w, txt, col, y=4, h=16): a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=col, ec="k", lw=0.6, alpha=0.9)); a.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=6.3)
    box(0.5, 17, "spin configuration\n$\\{\\hat n_i\\}$ on the lattice\n($S^2$ field)", "#E8F1FA"); box(23, 17, "SO(3)-equivariant\nencoder\n(e3nn, fixed features\n+ cell network)", "#DCEBDD"); box(46, 22, "latent: irreps $l=0..3$\n(4 channels each)\nglobal $z_g\\sim N(0,I)$,\ncell $z_c\\sim N(z_g,\\tau_l^2)$\n(top-down prior)", "#FDEFD0")
    box(74, 25.5, "decoder\n$\\lambda_{lm}(z)$, $l=1..6$\n(unrestricted)\n$E(\\hat n)=-\\sum\\lambda_{lm}Y_{lm}(\\hat n)$\n$p(\\hat n)\\propto e^{-E}$", "#F6DDE4")
    for x0, x1 in ((18.3, 22.2), (40.8, 45.2), (68.8, 73.2)): a.add_patch(FancyArrowPatch((x0, 12), (x1, 12), arrowstyle="-|>", mutation_scale=8, lw=0.9, color="k"))
    a.text(50, 0.8, "likelihood per site: $\\lambda\\cdot Y(\\hat n_i)-\\log Z(\\lambda)$, $\\log Z$ by Lebedev quadrature;   KL to the hierarchical prior", ha="center", fontsize=6.3); panel(a, "a", dx=0.0, dy=0.88)
    # (b) three site distributions
    Y = None
    from scipy.special import eval_legendre
    dens = [(lambda V: 20 * V[..., 2], "$l=1$: one cap\n(vMF, $\\kappa=20$)", "dipole (polar order)"), (lambda V: 4 * eval_legendre(2, V[..., 2]), "$l=2$: two caps\n(Maier–Saupe, $a_2=4$)", "nematic order"),
        (lambda V: np.log(sum(np.exp(-0.5 * (np.angle(np.exp(1j * (np.arctan2(V[..., 1], V[..., 0]) - 2 * np.pi * k / 3))) / 0.1) ** 2 - 0.5 * (np.arcsin(np.clip(V[..., 2], -1, 1)) / 0.3) ** 2) for k in range(3)) + 1e-300), "$l=3$: three coplanar caps\n(D$_{3h}$; widths 0.1 in / 0.3 out of plane)", "octupolar order")]
    peaks = []
    for k, (f, ttl, sub) in enumerate(dens):
        ax = fig.add_subplot(gs[1, k], projection="mollweide"); lon = np.linspace(-np.pi, np.pi, 100); lat = np.linspace(-np.pi / 2, np.pi / 2, 50); LON, LAT = np.meshgrid(lon, lat); V = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)], -1); lp = f(V); p = np.exp(lp - lp.max()); w = np.cos(LAT); p = p / (p * w).sum() / ((lon[1] - lon[0]) * (lat[1] - lat[0])); peaks.append(p.max())
        im = ax.pcolormesh(LON, LAT, p, shading="auto", cmap="viridis", vmin=0, vmax=max(peaks) if k == 2 else None, rasterized=True); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=6.0, pad=2); ax.text(0.5, -0.12, sub, transform=ax.transAxes, ha="center", fontsize=6.5, fontweight="bold")
        if k == 0: panel(ax, "b", dx=0.0, dy=1.18)
    savefig(fig, "fig01")
    MP["fig01"] = dict(title="Concept and architecture", panels={"a": "schematic (hand-drawn in make_figures code)", "b": "analytic densities (no data file); colour scale normalised per panel (density integrates to 1 on the sphere); each panel's own colour range"})
def _oracle(): return load("results/oracle.json")
def fig04():
    from analytic_ms import ms_curve
    fig = plt.figure(figsize=(W2, 5.4)); gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.38, wspace=0.55)
    ORA = _oracle(); EV = load("results/eval.json"); a = fig.add_subplot(gs[0, 0:2]); Ts = [r["T"] for r in ORA]; S = [r["S"] for r in ORA]
    a.plot(Ts, S, "o", color=OI["blue"], label="MC (L=12, global $Q$ frame)"); tt, ss = ms_curve(); tt = np.append(tt, tt[-1]); ss = np.append(ss, 0.0); a.plot(tt, ss, "-", color=OI["grey"], label="mean-field Maier–Saupe ($T_{NI}^{MF}=1.32J$)"); a.axvline(1.123, color="k", ls=":", lw=0.8); a.text(1.118, 0.05, "lit. $T_{NI}=1.123J$", fontsize=6, rotation=90, va="bottom", ha="right")
    a.set_xlabel("$T/J$"); a.set_ylabel("nematic order $S$"); a.legend(frameon=False, fontsize=6, loc="lower left"); a.set_xlim(0.75, 1.35); panel(a, "a")
    # (c) latent power and (d) KL
    c = fig.add_subplot(gs[0, 2]); runs = [r for r in EV if r["R"] == 1 and r["beta"] == 1.0 and r["alpha"] == 1e-5]; runs.sort(key=lambda r: (r["T"], r["st"]))
    for l, col in zip(range(4), (OI["grey"], OI["verm"], OI["blue"], OI["green"])): c.plot([r["T"] for r in runs], [max(r["latent_power_frac"][l], 1e-6) for r in runs], "o-", color=col, ms=3, label=f"$l={l}$")
    c.set_yscale("log"); c.set_ylim(1e-5, 2); c.set_xlabel("$T/J$"); c.set_ylabel("latent power fraction"); c.legend(frameon=False, fontsize=5.8, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 0.05)); c.axvline(1.123, color="k", ls=":", lw=0.8); panel(c, "c")
    d = fig.add_subplot(gs[0, 3]); T2B = md_table("results/v2_table_stageB.md"); kl1 = [(num(r["T"]), num(r["KL"])) for r in T2B if int(num(r["R"])) == 1]
    d.plot([r["T"] for r in runs], [r["kl"] for r in runs], "o--", color=OI["grey"], ms=3, label="v1 (not converged)"); d.plot([k[0] for k in kl1], [k[1] for k in kl1], "s-", color=OI["blue"], ms=3.5, label="converged ($R{=}1$, v2)")
    d.plot([0.95, 1.05], [15.1, 14.0], "D", color=OI["orange"], ms=4, label="separated decoder (v4 B)"); d.axhspan(12, 15, color=OI["green"], alpha=0.25, lw=0); d.text(1.2, 16.5, "12–15", fontsize=5.5, color=OI["green"]); d.set_yscale("log"); d.set_xlabel("$T/J$"); d.set_ylabel("KL (nats)"); d.legend(frameon=False, fontsize=5.3, loc="lower left", bbox_to_anchor=(0.0, 0.0)); panel(d, "d")
    # (b) site distributions R=1 decoders in the director frame: oracle (data fit) on top, v1 R=1 VAE below
    from fitlib import Fitter, real_sh
    from evaluate import load as load_model, val_data
    from mvae import sh as sh_e3
    Tb = [(0.80, "o"), (1.05, "o"), (1.14, "r"), (1.30, "r")]; peak = {}; panels = []
    for j, (T, st) in enumerate(Tb):
        S_ = np.load(f"{ROOT}/data/ll_L12_T{T:.3f}_{st}.npy").astype(float)[-50:]; Q = np.einsum('cxyzi,cxyzj->cij', S_, S_) / 12 ** 3; w, V = np.linalg.eigh(Q); frames = V[:, :, [0, 1, 2]]; 
                # pooled in the per-config director frame (director = column 2 -> pole); fit L=6
        f = Fitter(6, order=59, alpha=1e-8).fit(np.einsum('cxyzi,cij->cxyzj', S_, V).reshape(-1, 3)[:, [0, 1, 2]]); lam = f.lam
        def lp_oracle(Vv, f=f): return real_sh(6, Vv.reshape(-1, 3))[0].dot(f.lam).reshape(Vv.shape[:-1])
        run = f"results/runs/T{T:.2f}{st}_R1_b1_a1e-05.pt"; ck = torch.load(f"{ROOT}/{run}", weights_only=False); a_ = ck["args"]; from mvae import MVAE
        m = MVAE(L=12, R=1); m.load_state_dict(ck["state"]); m.eval(); x = torch.tensor(S_[:1]).float()
        with torch.no_grad(): mu, lv = m.encode(x); lam_v = m.decode(mu)[0, 0].numpy()
        def lp_vae(Vv, lam_v=lam_v): return (sh_e3(range(1, 7), torch.tensor(Vv.reshape(-1, 3), dtype=torch.float32)).numpy() @ lam_v).reshape(Vv.shape[:-1])
        Fr = np.linalg.eigh(Q[0])[1]                                   # frame of the first config: columns ascending; director = last column
        Fp = np.stack([Fr[:, 0], Fr[:, 1], Fr[:, 2]], -1)
        for row, (lpf, lab) in enumerate(((lp_oracle, "data fit"), (lp_vae, "VAE decoder"))):
            panels.append((j, row, lpf, Fp, T))
    # draw 2 rows inside the lower grid cells using sub-gridspec
    for j in range(4):
        sub = gs[1, j].subgridspec(2, 1, hspace=0.05); vm = None
        for row in range(2):
            jj, rr, lpf, Fp, T = panels[2 * j + row]; ax = fig.add_subplot(sub[row], projection="mollweide"); im, pm = plot_sphere_density(ax, lpf, frame=None if row == 0 else Fp, vmax=vm if row == 1 else None); vm = pm if row == 0 else vm
            ax.set_title(f"$T={Tb[j][0]}$" if row == 0 else "", fontsize=6.5, pad=2)
            if j == 0: ax.text(-0.15, 0.5, ["data fit", "VAE $R{=}1$"][row], transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=6)
    fig.text(0.07, 0.49, "b", fontsize=9, fontweight="bold")
    savefig(fig, "fig04")
    MP["fig04"] = dict(title="Nematic overview", panels={"a": "results/oracle.json (S per T), analytic mean-field Maier-Saupe (src/analytic_ms_heatkernel.py)", "b": "data/ll_L12_T*_{o,r}.npy last 50 snapshots (multipole refit L=6, fitlib) and v1 R=1 VAE decoders results/runs/T*_R1_b1_a1e-05.pt at the first snapshot, mean-z decode, director frame; NOTE v1 models are not converged (a2 underestimated at low T, REPORT.md)", "c": "results/eval.json latent_power_frac (v1)", "d": "results/eval.json kl (v1), results/v2_table_stageB.md (converged R=1), REPORT_v4 task B (15.1, 14.0)"})

def fig05():
    fig, ax = plt.subplots(1, 2, figsize=(W1 * 1.9, 2.7)); a, b = ax
    T1 = md_table("results/table_E1.md"); T = np.array([num(r["T"]) for r in T1]); st = [r["st"] for r in T1]; ve = np.array([num(r["odd/even VAE"]) for r in T1]); orc = np.array([num(r["odd/even oracle"]) for r in T1])
    a.semilogy(T, ve, "o", color=OI["verm"], ms=3.5, label="VAE (v1 run)"); a.semilogy(T, orc, "s", color=OI["blue"], ms=3.5, label="oracle fit (data)");     a.axhline(1e-3, color="k", ls=":", lw=0.8); a.text(1.18, 1.3e-3, "criterion $10^{-3}$", fontsize=6)
    a.axvline(1.123, color=OI["grey"], ls="--", lw=0.7); a.set_xlabel("$T/J$"); a.set_ylabel("odd / even multipole power"); a.legend(frameon=False, fontsize=6, loc="upper left"); panel(a, "a")
    Ls = [1, 2, 4]; rows = {}
    for l in open(f"{ROOT}/results/v2_table_f_frame_oracle.md"):
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if len(c) >= 7 and c[0][:1].isdigit(): rows.setdefault(float(c[0]), []).append([float(x) for x in c[1:7]])
    for t, rr in sorted(rows.items()):
        if t not in COL_LL: continue
        v = rr[0]; vae = [max(v[0], 1e-5), max(v[2], 1e-5), max(v[4], 1e-5)]; orc_ = [max(v[1], 1e-5), max(v[3], 1e-5), max(v[5], 1e-5)]
        b.semilogy(Ls, vae, "o-", color=COL_LL[t], ms=3.5, label=f"$T={t:.2f}$"); b.semilogy(Ls, orc_, "x", color=COL_LL[t], ms=5)
    b.set_xticks([1, 2, 4]); b.set_xlabel("decoder order $L$ (refit in the decoder frame)"); b.set_ylabel(r"$\Delta$NLL vs $L=6$ (nats/site)"); b.legend(frameon=False, fontsize=5.5, title="lines: VAE frame, ×: oracle frame", title_fontsize=5.3, loc="upper right", ncol=2); panel(b, "b")
    fig.tight_layout(); savefig(fig, "fig05")
    MP["fig05"] = dict(title="Nematic symmetry verdict", panels={"a": "results/table_E1.md (odd/even VAE: v1 runs; oracle)", "b": "results/v2_table_f_frame_oracle.md (VAE-frame and oracle-frame refit, dNLL vs L=6)"})
