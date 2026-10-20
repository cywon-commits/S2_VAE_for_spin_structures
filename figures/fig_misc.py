"""Figures 2, 3 and supplementary S1-S6.  Data: figures/data/misc.json (compute_misc.py), figures/data/ll_derived.json."""
import sys, json, numpy as np
from style import *
from tables import md_table, num, nums
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/src/kag"); sys.path.insert(0, f"{ROOT}/src/layered"); sys.path.insert(0, f"{ROOT}/src/kagome")
MISC = json.load(open(f"{HERE}/data/misc.json")); LLD = json.load(open(f"{HERE}/data/ll_derived.json")); MP = {}
GRP = ["D∞h", "D3h", "D3d", "C3v", "C6v", "C∞v"]; GCOL = [OI["blue"], OI["verm"], OI["orange"], OI["green"], OI["sky"], OI["purple"]]
def fig02():
    fig = plt.figure(figsize=(W1, 3.3)); ax = fig.add_axes([0.12, 0.12, 0.85, 0.82]); w = 0.13; sym = MISC["sym"]
    for k, g in enumerate(GRP): ax.bar(np.arange(7) + (k - 2.5) * w, sym[g], w, color=GCOL[k], label=g)
    ax.set_xticks(range(7)); ax.set_xlabel("multipole order $l$"); ax.set_ylabel("allowed components $n_l(K)$"); ax.set_ylim(0, 6.2); ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper right", columnspacing=0.8, handlelength=1.0)
    ins = fig.add_axes([0.14, 0.62, 0.40, 0.24], projection="mollweide"); lon = np.linspace(-np.pi, np.pi, 120); lat = np.linspace(-np.pi / 2, np.pi / 2, 60); LON, LAT = np.meshgrid(lon, lat); V = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)], -1)
    f = V[..., 0] ** 3 - 3 * V[..., 0] * V[..., 1] ** 2; ins.pcolormesh(LON, LAT, f, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto", rasterized=True); ins.set_xticks([]); ins.set_yticks([]); ins.set_title(r"D$_{3h}$, $l{=}3$: Re$(n_x+in_y)^3$", fontsize=5.5, pad=1)
    savefig(fig, "fig02")
    MP["fig02"] = dict(title="Site selection rules", panels={"-": "figures/data/misc.json sym (compute_misc.py = sym_check.py; group-averaged representation traces; D_inf and C_inf approximated by D60/C60); matches Appendix A of REQUEST_FIGURES exactly"})
def fig03():
    fig = plt.figure(figsize=(W2, 5.0)); gs = fig.add_gridspec(2, 6, hspace=0.5, wspace=1.6)
    # (a) logZ error and required degree
    a = fig.add_subplot(gs[0, 0:2]); E = MISC["logz_err"]; ks = np.array(E["kappa"], float)
    for o, col in zip(E["orders"], (OI["sky"], OI["blue"], OI["green"], OI["verm"])): e = np.maximum(np.array(E["err"][str(o)]), 1e-8); a.loglog(ks, e, "o-", color=col, ms=3, label=f"order {o}")
    a.axhline(1e-3, color="k", ls=":", lw=0.7); a.set_xlabel(r"$\kappa$ (vMF)"); a.set_ylabel(r"$|\Delta\log Z|$"); a.set_ylim(1e-8, 20); a.legend(frameon=False, fontsize=5.5, loc="upper left"); panel(a, "a", dx=0.0)
    a2 = fig.add_subplot(gs[0, 2:4]); N_ = MISC["logz_need"]; kk = np.array(N_["kappa"], float); a2.loglog(kk, N_["degree"], "o", color=OI["blue"], ms=4, label="needed for $10^{-3}$"); kx = np.logspace(0, 2.4, 50); a2.loglog(kx, 3.5 * np.sqrt(kx) + 3, "-", color=OI["verm"], label=r"$3.5\sqrt{\kappa}+3$")
    a2.set_xlabel(r"$\kappa$"); a2.set_ylabel("Lebedev degree"); a2.minorticks_off(); a2.legend(frameon=False, fontsize=5.8)
    # (b) support problem
    b = fig.add_subplot(gs[0, 4:6]); T3 = MISC["toy3"]; al = np.array([r["alpha"] for r in T3])
    for l, col in zip(range(1, 9), plt.cm.viridis(np.linspace(0, 0.95, 8))): b.semilogx(al, [r["P"][str(l)] for r in T3], "-", color=col, lw=0.9, label=f"$l={l}$")
    b.set_xlabel(r"regularisation $\alpha$"); b.set_ylabel("power fraction $P_l$"); b.legend(frameon=False, fontsize=5, ncol=2, loc="upper left"); b2 = b.twinx(); b2.semilogx(al, [r["nll_excess"] for r in T3], "k--", lw=1.0); b2.set_ylabel("NLL excess (nats, dashed)"); b2.spines["right"].set_visible(True); panel(b, "b", dx=0.0)
    # (c) estimators
    c = fig.add_subplot(gs[1, 0:3]); T1 = MISC["toy1"]
    for s2, mk in ((0.01, "o"), (0.04, "s")):
        rr = [r for r in T1 if r["s2_true"] == s2]; c.plot([r["a2"] for r in rr], [r["s2_moment"] for r in rr], mk + "-", color=OI["blue"], ms=4, label=f"moment estimator ($\\sigma^2_{{true}}={s2}$)"); c.plot([r["a2"] for r in rr], [r["s2_from_a4"] for r in rr], mk + "--", color=OI["orange"], mfc="none", ms=4, label=r"$-\frac{35}{18}a_4/a_2^2$" + f" ($\\sigma^2_{{true}}={s2}$)"); c.axhline(s2, color="k", lw=0.6, ls=":")
    c.set_xscale("log"); c.set_xticks([2, 4, 8, 15]); c.set_xticklabels(["2", "4", "8", "15"]); c.minorticks_off(); c.set_xlabel(r"$a_2$"); c.set_ylabel(r"estimated $\sigma^2$"); c.legend(frameon=False, fontsize=5.4, loc="lower left"); c.set_ylim(-0.1, 0.07); panel(c, "c", dx=0.0)
    # (d) mixing bias
    d = fig.add_subplot(gs[1, 3:6]); mix = LLD["mix"]
    for a0, s2, ls_ in ((4.0, 0.03, "-"), (3.0, 0.01, "--")):
        rr = sorted([m for m in mix if m["a0"] == a0 and m["s2"] == s2], key=lambda m: m["r"]); d.plot([m["r"] for m in rr], [m["per_sample"] - s2 for m in rr], "o" + ls_, color=OI["blue"], ms=3.5, label=f"per-sample ($a_0={a0:g}$, $\\sigma^2={s2}$)"); d.plot([m["r"] for m in rr], [m["avg_moment"] - s2 for m in rr], "s" + ls_, color=OI["verm"], ms=3.5, mfc="none", label=f"averaged moments ($a_0={a0:g}$, $\\sigma^2={s2}$)")
    d.axhline(0, color="k", lw=0.6); d.set_xlabel("relative spread $r$ of the amplitude $a$"); d.set_ylabel(r"$\hat\sigma^2-\sigma^2_{true}$"); d.legend(frameon=False, fontsize=5.0, loc="lower left"); d.set_ylim(-0.06, 0.003); panel(d, "d", dx=0.0)
    savefig(fig, "fig03")
    MP["fig03"] = dict(title="Method checks", panels={"a": "misc.json logz_err, logz_need (compute_misc.py = logz.py; vMF exact log Z; degree for 1e-3)", "b": "misc.json toy3 (toy3.py: ring theta0=60deg, width 0.15, N=3e5, L=8 refit)", "c": "misc.json toy1 (toy1.py: sample_bingham + small random rotations, N=4e5, Fitter L=8) and the moment estimator of moment_est.py", "d": "ll_derived.json mix (mix_check.py)"})
def fig_s1():
    S = MISC["samplers"]; fig, ax = plt.subplots(2, 4, figsize=(W2, 3.6)); ax = ax.ravel(); names = [("linear_k3", r"linear field $h=3$"), ("linear_k20", r"linear $h=20$"), ("uni_K-3.0_h1.0", r"uniaxial $K=-3,h_z=1$"), ("uni_K3.0_h1.0", r"$K=3,h_z=1$"), ("uni_K8.0_h0.5", r"$K=8,h_z=0.5$"), ("uni_K8.0_h-12.0", r"$K=8,h_z=-12$"), ("bingham_a15", r"Bingham, $a=15$ ($n_z$)")]
    xg = np.linspace(-1, 1, 401)
    for a, (k, t) in zip(ax[:7], names): h = S[k]["h"]; a.plot(h["centers"], h["dens"], "o", color=OI["blue"], ms=2.5, label="exact sampler"); a.plot(xg, S[k]["exact"], "-", color=OI["verm"], lw=1.0, label="exact density"); a.set_title(t, fontsize=6.5); a.set_xlabel(r"$\cos\theta$", fontsize=7)
    ax[0].legend(frameon=False, fontsize=5.5); a = ax[7]; Ks = S["acc_chord_K"]
    for k_, ls in (("hz0", "-"), ("hz5", "--"), ("hz-30", ":")): a.semilogx(Ks, S["acc_chord"][k_], ls, color=OI["blue"], label=f"chord, $h_z={k_[2:]}$")
    a.semilogx(S["acc_bingham_a"], S["acc_bingham"], "o-", color=OI["verm"], ms=3, label="Bingham (ACG)"); a.set_xlabel("$K$ or $a$"); a.set_ylabel("acceptance"); a.set_ylim(0, 1.05); a.legend(frameon=False, fontsize=5.2)
    fig.tight_layout(); savefig(fig, "figS1"); MP["figS1"] = dict(title="Exact single-site samplers", panels={"-": "misc.json samplers (compute_misc.py: samplers.py + test_samplers.py + acc.py procedures; N=4e5)"})
def fig_s2():
    import analytic_ms as ms
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.4)); T1 = MISC["toy1"]; a = ax[0]; pred = np.array([-18 / 35 * r["a2"] ** 2 * r["s2_true"] for r in T1]); fit = np.array([r["a4_fit"] for r in T1])
    for r, p, f_ in zip(T1, pred, fit): a.plot(p, f_, "o" if r["a2"] <= 4 else "x", color=OI["blue"] if r["a2"] <= 4 else OI["verm"], ms=4)
    lim = [-0.8, 0.1]; a.plot(lim, lim, "k--", lw=0.7); a.set_xlim(lim); a.set_xlabel(r"predicted $a_4=-\frac{18}{35}a_2^2\sigma^2$"); a.set_ylabel(r"fitted $a_4$ (L=8)"); a.set_ylim(-0.8, 0.1); a.text(0.03, 0.97, "● $a_2\\leq4$ (valid)\n$a_2\\geq8$: outside the plot\n(fit not valid, see Fig. 3c)", transform=a.transAxes, fontsize=5.6, va="top"); panel(a, "a")
    tt, ss = ms.ms_curve(); tt = tt / 6; b = ax[1]; b.plot(tt, ss, color=OI["blue"]); b.set_xlabel(r"$t=T/(zJ)$"); b.set_ylabel("mean-field $S$"); b.axvline(0.2202, color="k", ls=":", lw=0.7); b.text(0.2185, 0.05, "$t_{NI}=0.2202$", fontsize=6, rotation=90, ha="right"); panel(b, "b")
    c = ax[2]; c.plot(tt, ss / tt, color=OI["blue"], label=r"mean-field $a_2=S/t$"); ORA = load("results/oracle.json"); c.plot([r["T"] / 6 for r in ORA if r["T"] <= 1.1], [r["a2"] for r in ORA if r["T"] <= 1.1], "o", color=OI["verm"], ms=3.5, label=r"MC fit $a_2$ (smeared)"); c.set_xlabel(r"$t=T/(zJ)$"); c.set_ylabel("$a_2$"); c.legend(frameon=False, fontsize=6); panel(c, "c")
    fig.tight_layout(); savefig(fig, "figS2"); MP["figS2"] = dict(title="Analytic calculations", panels={"a": "misc.json toy1", "b,c": "src/analytic_ms.py (= analytic_ms_heatkernel.py section 1); MC points: results/oracle.json"})
def fig_s3():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4)); T4 = MISC["toy4"]; Ls = [1, 2, 3, 4, 6, 8, 10, 12]; ref = T4["nll"]["12"]; a = ax[0]; a.semilogy(Ls, [max(T4["nll"][str(L)] - ref, 1e-4) for L in Ls], "o-", color=OI["blue"]); a.set_xticks(Ls); a.set_xlabel("order $L$"); a.set_ylabel("NLL excess over $L=12$ (nats)"); a.annotate("big drop $L=2\\to3$", (3, T4["nll"]["3"] - ref), (5, 1.5), fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.6)); panel(a, "a")
    b = ax[1]; b.semilogx([r["alpha"] for r in T4["cusp"]], [r["c6_over_c3"] for r in T4["cusp"]], "o-", color=OI["blue"], label="fitted $E(\\phi)$"); b.axhline(-0.25, color=OI["verm"], ls="--", label="cusp prediction $-1/4$"); b.set_xlabel(r"regularisation $\alpha$"); b.set_ylabel(r"$c_6/c_3$ (equatorial Fourier of $E$)"); b.legend(frameon=False, fontsize=6); panel(b, "b")
    fig.tight_layout(); savefig(fig, "figS3"); MP["figS3"] = dict(title="Kagome three-cap toy", panels={"-": "misc.json toy4 (toy4.py: three caps, in-plane width 0.15, out-of-plane 0.35, N=4e5)"})
def fig_s4():
    P = MISC["pyrochlore"]; fig, ax = plt.subplots(figsize=(W1, 2.5)); x0 = 0
    for l in ("2", "4", "6"):
        for m, v in sorted(P[l].items(), key=lambda kv: int(kv[0])): ax.bar(x0, v, 0.8, color={"2": OI["blue"], "4": OI["orange"], "6": OI["green"]}[l]); ax.text(x0, v * 1.3, f"{v:.3g}", ha="center", fontsize=5.5, rotation=90); x0 += 1
        x0 += 0.6
    ax.set_yscale("log"); ax.set_ylim(1e-3, 30); labs = []; 
    for l in ("2", "4", "6"): labs += [f"$l={l}$\n$m={m}$" if i == 0 else f"$m={m}$" for i, m in enumerate(sorted(P[l], key=int))]
    xs = []; x0 = 0
    for l in ("2", "4", "6"): xs += list(range(int(x0), int(x0) + len(P[l]))); x0 += len(P[l]) + 0.6
    pos = []; x0 = 0
    for l in ("2", "4", "6"):
        for i in range(len(P[l])): pos.append(x0); x0 += 1
        x0 += 0.6
    ax.set_xticks(pos); ax.set_xticklabels([t.replace("$m=", "|m|=$") if False else t for t in labs], fontsize=5.5); ax.set_ylabel(r"power $\sum|c_{lm}|^2$ of $G_l(\hat n)$ at $|m|$"); fig.tight_layout(); savefig(fig, "figS4")
    MP["figS4"] = dict(title="Pyrochlore geometry", panels={"-": "misc.json pyrochlore (analytic_pyrochlore_skl.py section 3; matches Appendix A: l=4 |m|=3: 4.290, m=0: 0.00192)"})
def skl_texture(m0, kind, N=160):
    q = 1.0; qs = [q * np.array([np.cos(2 * np.pi * a / 3), np.sin(2 * np.pi * a / 3)]) for a in range(3)]; Q = np.array(qs[:2]); A = 2 * np.pi * np.linalg.inv(Q).T; u = np.linspace(0, 1, N, endpoint=False); U, V = np.meshgrid(u, u, indexing="ij"); R = U[..., None] * A[0] + V[..., None] * A[1]; m = np.zeros(R.shape[:2] + (3,)); m[..., 2] += m0
    for a in range(3):
        qh = qs[a] / np.linalg.norm(qs[a]); ph = R @ qs[a] + np.pi; e = np.array([-qh[1], qh[0], 0]) if kind == "bloch" else np.array([qh[0], qh[1], 0]); m += np.cos(ph)[..., None] * np.array([0, 0, 1.0]) + np.sin(ph)[..., None] * e
    m /= np.linalg.norm(m, axis=-1)[..., None]; return R, m
def fig_s5():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.4)); R, m = skl_texture(0.5, "bloch"); a = ax[0]; im = a.pcolormesh(R[..., 0], R[..., 1], m[..., 2], cmap="RdBu_r", shading="auto", rasterized=True); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_title(r"Bloch, $m_0=0.5$: $n_z$", fontsize=6.5, pad=10); panel(a, "a")
    b = ax[1]; S = MISC["skl"]
    for kind, mk, col in (("bloch", "o", OI["blue"]), ("neel", "s", OI["verm"])): rr = [r for r in S if r["kind"] == kind]; b.plot([r["m0"] for r in rr], [r["Y66"] for r in rr], mk + "-", color=col, label=kind.capitalize())
    b.axhline(0, color="k", lw=0.6); b.set_xlabel("$m_0$ (uniform magnetisation)"); b.set_ylabel(r"$\langle Y_{66}\rangle$"); b.legend(frameon=False, fontsize=6); panel(b, "b")
    c = ax[2]; r0 = [r for r in S if r["kind"] == "bloch" and r["m0"] == 0.5][0]; al = [v for k, v in r0["Ylm_abs"].items() if int(k.split(",")[1]) % 6 == 0]; fo = [v for k, v in r0["Ylm_abs"].items() if int(k.split(",")[1]) % 6 != 0]
    c.semilogy(np.arange(len(fo)), np.maximum(sorted(fo, reverse=True), 1e-17), "o", color=OI["verm"], ms=3, label=r"$m\not\equiv0$ (mod 6): forbidden"); c.semilogy(np.arange(len(al)) + 0.0, np.maximum(sorted(al, reverse=True), 1e-17), "s", color=OI["blue"], ms=4, label=r"$m\equiv0$ (mod 6)"); c.set_xlabel("component (sorted)"); c.set_ylabel(r"$|\langle Y_{lm}\rangle|$, $m\geq0$, $l\leq7$"); c.legend(frameon=False, fontsize=5.5); c.set_ylim(1e-18, 3); panel(c, "c")
    fig.tight_layout(); savefig(fig, "figS5"); MP["figS5"] = dict(title="Triple-q skyrmion lattice moments", panels={"a": "texture recomputed in figure code (same construction as analytic_pyrochlore_skl.py)", "b,c": "misc.json skl (Q=-1 for all; <Y66> Bloch -0.0222/-0.0117/-0.0063 for m0=0/0.5/1, Neel opposite sign = Appendix A)"})
def fig_s6():
    from cfg_terms import block_frames
    fig = plt.figure(figsize=(W2, 3.0)); gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.9], wspace=0.28); a = fig.add_subplot(gs[0]); a.axis("off"); a.set_xlim(0, 12); a.set_ylim(0, 8)
    rows = [("none", "trivial: frame returns to itself", OI["green"]), (r"Z$_2$", "returns up to $-1$ in the double cover", OI["sky"]), (r"C$_3$", "triad relabelled cyclically (1/3 vortex)", OI["orange"]), (r"C$_2$", "plane normal reversed (1/2 dislocation)", OI["verm"])]
    a.text(0, 7.4, "classification in the binary dihedral group $2D_3$", fontsize=6.8, fontweight="bold")
    for i, (n, d, col) in enumerate(rows): y = 6.2 - 1.55 * i; a.add_patch(plt.Rectangle((0, y - 0.45), 1.5, 1.0, color=col, alpha=0.5, lw=0)); a.text(0.75, y + 0.05, n, ha="center", va="center", fontsize=7); a.text(1.7, y + 0.05, d, va="center", fontsize=5.8)
    a.text(0, 0.2, "Frame per block: $\\nu$ = min. eigenvector of $Q$, triad phase $\\arg\\langle e^{3i\\phi}\\rangle/3$;\ncandidates $\\pm$, 3 cyclic phases; lift along the loop of 4 blocks.", fontsize=5.8); panel(a, "a", dx=0.0, dy=0.95)
    b = fig.add_subplot(gs[1]); L = 12; x, y = np.meshgrid(np.arange(L), np.arange(L), indexing="ij"); res = {}
    def colour_state(kind):
        S = np.zeros((L, L, 3, 3)); u = [np.array([np.cos(2 * np.pi * k / 3), np.sin(2 * np.pi * k / 3), 0.0]) for k in range(3)]
        for s in range(3):
            col = (np.zeros((L, L), int) + s) % 3 if kind == "q0" else (2 * x + y + s) % 3
            for k in range(3): S[..., s, :][col == k] = u[k]
        return S
    from compute_kag_defects import classify_blocks_count
    rng = np.random.default_rng(0)
    for kind in ("q0", "sqrt3"):
        for sig in (0.0, 0.03, 0.08):
            S = colour_state(kind); cnts = []
            for _ in range(20):
                Sn = S + sig * rng.standard_normal(S.shape); Sn /= np.linalg.norm(Sn, axis=-1, keepdims=True); cnts.append(classify_blocks_count(Sn, 3))
            res[(kind, sig)] = np.mean(cnts)
    xs = np.arange(3); w = 0.35
    for k, (kind, col) in enumerate((("q0", OI["blue"]), ("sqrt3", OI["orange"]))): b.bar(xs + (k - 0.5) * w, [res[(kind, s)] for s in (0.0, 0.03, 0.08)], w, color=col, label=r"$q=0$" if kind == "q0" else r"$\sqrt{3}\times\sqrt{3}$")
    b.set_xticks(xs); b.set_xticklabels(["0", "0.03", "0.08"]); b.set_xlabel("Gaussian noise on the spins"); b.set_ylabel("defects per configuration (3x3 blocks)"); b.set_ylim(0, 1); b.legend(frameon=False, fontsize=6, loc="upper right")
    for k in range(2):
        for i, sg in enumerate((0.0, 0.03, 0.08)): b.text(i + (k - 0.5) * w, 0.03, f"{[res[('q0', sg)], res[('sqrt3', sg)]][k]:.2f}", ha="center", fontsize=6)
    panel(b, "b")
    savefig(fig, "figS6"); MP["figS6"] = dict(title="Defect classification", panels={"a": "schematic", "b": "classifier validation computed in figure code on ideal q=0 and sqrt3xsqrt3 coplanar states (+ Gaussian noise), 20 draws each; expected 0 defects"})
