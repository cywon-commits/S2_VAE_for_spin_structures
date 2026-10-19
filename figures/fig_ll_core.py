"""Figures 6, 7, 8 (nematic core figures).  Data: figures/data/ll_derived.json (compute_ll.py, recomputed on data/ll_L12_T*_o.npy) and results/*.md,*.json tables of the earlier reports."""
import numpy as np, json
from style import *
from tables import md_table, num, nums
D = json.load(open(f"{HERE}/data/ll_derived.json")); TS = [0.80, 0.95, 1.05, 1.10]; ORA = load("results/oracle.json")
TC = md_table("results/v4_table_C.md"); TA = md_table("results/v4_table_A.md"); T2B = md_table("results/v2_table_stageB.md"); T5 = md_table("results/v5_table_13_14.md")
def tc(T, R, key): return next(num(r[key]) for r in TC if abs(num(r["T"]) - T) < 1e-6 and int(num(r["R"])) == R)
def ta(T, R, key): return next(r[key] for r in TA if abs(num(r["T"]) - T) < 1e-6 and int(num(r["R"])) == R)
def t2b(T, R, key): return next(r[key] for r in T2B if abs(num(r["T"]) - T) < 1e-6 and int(num(r["R"])) == R)
MP = {}                                                            # manifest notes
def fig06():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.35)); m = D["moments"]
    # (a) a_loc vs mean field
    a = ax[0]; lim = [1.8, 6.2]; a.plot(lim, lim, color=OI["grey"], lw=0.8, ls="--", label=r"$a_{loc}=\beta zJS$"); a.fill_between(lim, [l * 0.95 for l in lim], [l * 1.05 for l in lim], color=OI["grey"], alpha=0.15, lw=0, label=r"$\pm5\%$")
    for T in TS:
        mm = m[str(T)]; a.errorbar(mm["a_mf"], mm["a_loc"], xerr=mm["a_mf_se"], yerr=mm["a_loc_se"], fmt="*", ms=7, color=COL_LL[T], mec="k", mew=0.4, zorder=5, label="data (oracle)" if T == 0.80 else None)
        for R in (1, 2, 3, 4): a.plot(6 * 0 + m[str(T)]["a_mf"] + 0.0, tc(T, R, "a_loc"), MARK_R[R], color=COL_LL[T], mfc="none" if R > 1 else COL_LL[T], ms=4, label=f"VAE R={R}" if T == 0.80 else None)
    a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel(r"mean-field $\beta zJS=6S/T$"); a.set_ylabel(r"local field $a_{loc}$ (moment estimator)"); a.legend(frameon=False, loc="upper left", fontsize=6); panel(a, "a")
    # (b) a2 (smeared fit), a_loc, mean field vs T
    b = ax[1]; Tt = np.array([r["T"] for r in ORA if r["T"] <= 1.10]); a2 = np.array([r["a2"] for r in ORA if r["T"] <= 1.10]); mf = np.array([r["a2_MF"] for r in ORA if r["T"] <= 1.10])
    b.plot(Tt, mf, "-", color=OI["grey"], label=r"mean field $\beta zJS$"); b.plot(Tt, a2, "o-", color=OI["verm"], label=r"fitted $a_2$ (smeared)"); b.errorbar(TS, [m[str(T)]["a_loc"] for T in TS], yerr=[m[str(T)]["a_loc_se"] for T in TS], fmt="s", color=OI["blue"], mfc="none", label=r"$a_{loc}$ (unsmeared)")
    b.set_xlabel(r"$T/J$"); b.set_ylabel(r"$a$"); b.legend(frameon=False, fontsize=6); panel(b, "b")
    # (c) sigma^2_site vs T
    c = ax[2]; c.errorbar(TS, [m[str(T)]["sigma2_site"] for T in TS], yerr=[m[str(T)]["sigma2_site_se"] for T in TS], fmt="o-", color=OI["blue"], label="data: moment estimator")
    c.plot(Tt, [r["sigma2"] for r in ORA if r["T"] <= 1.10], "s--", color=OI["orange"], mfc="none", label=r"data: $-\frac{35}{18}a_4/a_2^2$"); c.plot(TS, [tc(T, 1, "sigma2 (cell-axes frame oracle)") for T in TS], "^", color=OI["green"], mfc="none", label="VAE R=1 (frame oracle)")
    c.set_xlabel(r"$T/J$"); c.set_ylabel(r"$\sigma^2_{site}$ (rad$^2$)"); c.legend(frameon=False, fontsize=6); c.set_ylim(0, 0.1); panel(c, "c")
    fig.tight_layout(w_pad=1.0); savefig(fig, "fig06")
    MP["fig06"] = dict(title="Local mean field and smearing", panels={"a": "figures/data/ll_derived.json:moments (compute_ll.py on data/ll_L12_T*_o.npy, 5-block s.e.); VAE points: results/v4_table_C.md a_loc column (frame-oracle a_loc; plotted at the data mean-field abscissa of the same T)", "b": "results/oracle.json (a2 fit, a2_MF), ll_derived.json moments", "c": "ll_derived.json moments.sigma2_site; results/oracle.json sigma2; results/v4_table_C.md"})
def fig07():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.4)); c = D["cells"]; m = D["moments"]; Rs = [1, 2, 3, 4]
    a = ax[0]
    for T in TS:
        s2site = m[str(T)]["sigma2_site"]; pred = [s2site] + [max(s2site - c[str(T)][str(R)]["sigma2_cell"], 0) for R in (2, 3, 4)]
        a.plot(Rs, pred, "-", color=COL_LL[T], lw=1.0); a.plot(Rs, [tc(T, R, "sigma2 (cell-axes frame oracle)") for R in Rs], "o", color=COL_LL[T], label=f"$T={T:.2f}$")
    a.set_xticks(Rs); a.set_xlabel("latent resolution $R$"); a.set_ylabel(r"$\sigma^2_{VAE}(R)$ (rad$^2$)"); from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL_LL[T], marker="o", ls="-", ms=3.5, lw=1) for T in TS]; a.legend(h, [f"$T={T:.2f}$" for T in TS], frameon=False, fontsize=6, ncol=2, loc="upper right", title="line: prediction, dots: VAE", title_fontsize=6); panel(a, "a")
    b = ax[1]; ns = np.array([216, 64, 27])
    for T in TS:
        y = np.array([c[str(T)][str(R)]["sigma2_cell"] for R in (2, 3, 4)]); e = np.array([c[str(T)][str(R)]["sigma2_cell_se"] for R in (2, 3, 4)]); b.errorbar(ns, y, yerr=e, fmt="o-", color=COL_LL[T])
    b.set_xscale("log"); b.set_yscale("log"); b.invert_xaxis(); b.set_xlabel("cell size $n$ (sites)"); b.set_ylabel(r"$\sigma^2_{cell}$ (shuffle-corrected)"); b.set_xticks(ns); b.set_xticklabels([str(n) for n in ns]); b.minorticks_off(); panel(b, "b")
    cc = ax[2]; N = 1728
    for T in TS:
        cells = np.array([8, 27, 64]); pred = np.array([cells[k] * c[str(T)][str(R)]["G"] / N for k, R in enumerate((2, 3, 4))]); orc = np.array([c[str(T)][str(R)]["gain_net"] for R in (2, 3, 4)])
        cc.plot([2, 3, 4], pred, "-", color=COL_LL[T], lw=1.0); cc.plot([2, 3, 4], orc, "x", color=COL_LL[T], ms=5)
        cc.plot([2, 3, 4], [num(ta(T, R, "NLL gain")) for R in (2, 3, 4)], "o", color=COL_LL[T], label=f"$T={T:.2f}$")
    cc.set_xticks([2, 3, 4]); cc.set_xlabel("latent resolution $R$"); cc.set_ylabel("NLL gain vs $R=1$ (nats/site)"); h = [Line2D([], [], color=COL_LL[T], marker="o", ls="-", ms=3.5, lw=1) for T in TS] + [Line2D([], [], color="k", ls="-", lw=1), Line2D([], [], color="k", marker="x", ls="", ms=4), Line2D([], [], color="k", marker="o", ls="", ms=3.5)]; cc.legend(h, [f"$T={T:.2f}$" for T in TS] + [r"pred. $G=\frac{1}{2}nI\sigma^2_{cell}$", "cell-frame oracle", "VAE (up/down)"], frameon=False, fontsize=5.8, loc="upper left", ncol=1); cc.set_ylim(0.02, 0.24); panel(cc, "c")
    fig.tight_layout(w_pad=1.0); savefig(fig, "fig07")
    MP["fig07"] = dict(title="Latent resolution measured by smearing", panels={"a": "VAE: results/v4_table_C.md (sigma2, cell-axes frame oracle); line: ll_derived.json moments.sigma2_site minus cells.sigma2_cell", "b": "ll_derived.json cells (budget.py procedure, 100 configs, shuffle control, 5-block s.e.)", "c": "line: ll_derived.json cells.G (G=n I sigma2_cell/2, I from oracle a2) x cells/1728; x: ll_derived.json cells.gain_net (cell_gain.py procedure); points: results/v4_table_A.md NLL gain column"})
def univ(G, d): G = np.asarray(G, float); return 1 - (d / (2 * G)) * np.log1p(2 * G / d)
def kag_channel_G():
    """Kagome channel gains per unit (data side): local frame (A->B, per 3x3 cell), ring rotation (B->C, per loop, overfit subtracted), defect (regression b, per defect).  Source: results/layered_gain*.txt, results/kag/regress_k6u*, REPORT_E3_v2."""
    return dict(local=190 / 16.0, ring=(89 - 24) / 47.5, defect=15.3)
def fig08():
    c = D["cells"]; amp = D["amp"]; N = 1728; ncell = {2: 8, 3: 27, 4: 64}; nsite = {2: 216, 3: 64, 4: 27}
    fig, ax = plt.subplots(2, 2, figsize=(W2, 5.6)); (a, b), (cx, d) = ax
    # (a) channel gains stacked + VAE points
    bw = 0.34; xs = []; lab = []; pos = 0
    biax = {2: 0.004, 3: 0.013, 4: 0.025}                                   # NLL gain of the biaxial deviation (v4 subspace minus v5 1.3 ablation, T=1.05; REPORT_v5 sec 1.3)
    for T in (0.95, 1.05):
        for R in (2, 3, 4):
            g_dir = c[str(T)][str(R)]["gain_net"]; g_amp = amp[str(T)][str(R)]["gain"]; g_bi = biax[R] if T == 1.05 else 0.0; x0 = pos
            a.bar(x0, g_dir, bw * 1.6, color=COL_CH["director"], label="director" if pos == 0 else None); a.bar(x0, g_amp, bw * 1.6, bottom=g_dir, color=COL_CH["amp"], label="amplitude" if pos == 0 else None)
            if T == 1.05: a.bar(x0, g_bi, bw * 1.6, bottom=g_dir + g_amp, color=COL_CH["biax"], label="biaxial" if R == 2 else None)
            xs.append(x0); lab.append(f"{R}"); pos += 1
        pos += 0.7
    gv = [num(ta(T, R, "NLL gain")) for T in (0.95, 1.05) for R in (2, 3, 4)]; a.plot(xs, gv, "ko", ms=4, label="VAE (up/down prior)")
    g5 = [num(next(r["NLL gain vs R=1"] for r in T5 if int(num(r["R"])) == R and r["model"].startswith("1.4"))) for R in (2, 3, 4)]; a.plot(xs[3:], g5, "D", color="none", mec="k", ms=4, label="VAE (separated + subspace)")
    a.set_xticks(xs); a.set_xticklabels(lab); a.set_xlabel(r"$R$   (left block $T=0.95$, right block $T=1.05$)"); a.set_ylabel("NLL gain vs $R=1$ (nats/site)"); a.legend(frameon=False, fontsize=6, loc="upper left"); panel(a, "a")
    # (b) KL accounting at T=1.05
    Rr = [2, 3, 4]; w3 = 0.2; I = c["1.05"]["2"]["I"]; kl1 = 20.3; Gd = {R: c["1.05"][str(R)]["G"] for R in Rr}
    hier_dir = [ncell[R] * np.log1p(Gd[R]) for R in Rr]; hier_amp = [ncell[R] * 0.5 * np.log1p(2 * amp["1.05"][str(R)]["Ga"]) for R in Rr]; indep = [ncell[R] * kl1 * np.log(nsite[R] * I) / np.log(N * I) - kl1 for R in Rr]
    obs2 = [nums(ta(1.05, R, "v2 independent: gain / extra KL / net"))[1] for R in Rr]; obs4 = [num(ta(1.05, R, "extra KL")) for R in Rr]; obs5 = [num(next(r["extra KL vs R=1"] for r in T5 if int(num(r["R"])) == R and r["model"].startswith("1.4"))) for R in Rr]
    x = np.arange(3); b.bar(x - 1.5 * w3, np.array(hier_dir) + np.array(hier_amp), w3, color="none", edgecolor="k", hatch="////", label="hierarchical budget (dir + amp)"); b.bar(x - 0.5 * w3, indep, w3, color="none", edgecolor=OI["grey"], hatch="\\\\\\\\", label="independent-prior estimate")
    b.bar(x + 0.5 * w3, obs2, w3, color=OI["grey"], label="observed: independent (v2)"); b.bar(x + 1.5 * w3, obs4, w3, color=OI["sky"], label="observed: up/down (v4)"); b.bar(x + 2.5 * w3, obs5, w3, color=OI["blue"], label="observed: separated+subspace (v5)")
    b.set_xticks(x + 0.5 * w3); b.set_xticklabels(Rr); b.set_xlabel("$R$ ($T=1.05$)"); b.set_ylabel("extra KL vs $R=1$ (nats / config)"); b.legend(frameon=False, fontsize=5.6, loc="upper left"); panel(b, "b")
    # (c) universal curve
    G = np.logspace(-1.3, 2.0, 300)
    for dd, ls in ((1, ":"), (2, "-"), (3, "--")): cx.plot(G, univ(G, dd), ls, color=OI["grey"], lw=1.0); cx.text(70, univ(70, dd) + (0.03 if dd == 1 else -0.03 if dd == 3 else 0.0), f"$d={dd}$", fontsize=6, ha="right", va="center", color=OI["grey"])
    cx.plot(G, 1 - np.log1p(G) / G * 0 - 0 * G + 0 * G, alpha=0)                                     # keep limits
    for R, mk in zip(Rr, ("o", "s", "^")):
        cx.plot(Gd[R], univ(Gd[R], 2), mk, color=COL_CH["director"], label="director (nematic)" if R == 2 else None); cx.plot(amp["1.05"][str(R)]["Ga"], univ(amp["1.05"][str(R)]["Ga"], 1), mk, color=COL_CH["amp"], label="amplitude" if R == 2 else None)
        gb = nsite[R] * biax[R]; cx.plot(gb, univ(gb, 2), mk, color=COL_CH["biax"], label="biaxial" if R == 2 else None)
    kg = kag_channel_G(); cx.plot(kg["local"], univ(kg["local"], 3), "*", color=COL_CH["local"], ms=8, mec="k", mew=0.3, label="kagome local frame ($d=3$)"); cx.plot(kg["ring"], univ(kg["ring"], 1), "*", color=COL_CH["ring"], ms=8, mec="k", mew=0.3, label="kagome ring rotation")
    cx.plot(kg["defect"], univ(kg["defect"], 1), "*", color=COL_CH["defect"], ms=8, mec="k", mew=0.3, label="kagome defect ($d\\!=\\!1$ assumed)")
    for R, mk in zip(Rr, ("o", "s", "^")):                                                         # measured VAE net / NLL-gain (all channels, T=1.05, v5 1.4)
        ng = nums(next(r["net ELBO gain vs R=1"] for r in T5 if int(num(r["R"])) == R and r["model"].startswith("1.4")))[0]; gain = g5[R - 2] * N; cx.plot(gain / ncell[R], ng / gain, mk, mfc="none", mec="k", label="VAE measured (all channels)" if R == 2 else None)
    cx.plot(618.7 / 16, 417.6 / 618.7, "*", color="none", mec="k", ms=9, label="kagome VAE (ii)$-$(iv)")
    cx.set_xscale("log"); cx.set_xlabel(r"information gain per unit, $G$ (nats)"); cx.set_ylabel(r"net gain / $G$"); cx.set_ylim(-0.05, 1.0); cx.legend(frameon=False, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.2, columnspacing=0.8); panel(cx, "c")
    # (d) predicted vs achieved net ELBO gain
    pred = [num(ta(1.05, R, "predicted")) for R in Rr]; iso = [num(ta(1.05, R, "net ELBO gain")) for R in Rr]; sub = [nums(next(r["net ELBO gain vs R=1"] for r in T5 if int(num(r["R"])) == R and r["model"].startswith("subspace")))[0] for R in Rr]; v5 = [nums(next(r["net ELBO gain vs R=1"] for r in T5 if int(num(r["R"])) == R and r["model"].startswith("1.4")))[0] for R in Rr]
    w4 = 0.2; x = np.arange(3); d.bar(x - 1.5 * w4, pred, w4, color="none", edgecolor="k", hatch="////", label="predicted"); d.bar(x - 0.5 * w4, iso, w4, color=OI["grey"], label="isotropic (v4)"); d.bar(x + 0.5 * w4, sub, w4, color=OI["sky"], label="subspace cov. (v4)"); d.bar(x + 1.5 * w4, v5, w4, color=OI["blue"], label="separated + subspace (v5)")
    for k in range(3): d.text(x[k] + 1.5 * w4, v5[k] + 2, f"{100 * v5[k] / pred[k]:.0f}%", ha="center", fontsize=6)
    d.set_xticks(x); d.set_xticklabels(Rr); d.set_xlabel("$R$ ($T=1.05$)"); d.set_ylabel("net ELBO gain vs $R=1$ (nats / config)"); d.legend(frameon=False, fontsize=6, loc="upper left"); panel(d, "d")
    fig.tight_layout(); savefig(fig, "fig08")
    MP["fig08"] = dict(title="Information accounting", panels={"a": "ll_derived.json cells.gain_net (director), amp.gain (amplitude); biaxial: results/v4_table_A.md/v5_table_13_14.md (v4 subspace NLL gain minus 1.3 ablation: 0.004/0.013/0.025, REPORT_v5 sec 1.3, hard-coded); points: v4_table_A.md, v5_table_13_14.md",
        "b": "budgets: ll_derived.json (G, amp Ga) with the formulas of budget2.py/amp_gain.py; independent-prior estimate: budget2.py formula with R=1 KL=20.3; observed: results/v4_table_A.md (v2 indep., v4), v5_table_13_14.md (v5)", "c": "G values: ll_derived.json (nematic), results/layered_gain*.txt and results/kag/regress_k6u_T0.0020.json (kagome); measured VAE ratios: v5 table; kagome VAE (ii)-(iv): REPORT_E3_v4", "d": "results/v4_table_A.md, results/v5_table_13_14.md"})
