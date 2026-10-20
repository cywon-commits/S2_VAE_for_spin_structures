"""Figure 14 (REQUEST_FIGURES_v2 A2): the universal curve net/G = 1 - (d/2G) ln(1+2G/d) as a stand-alone two-column figure (former Fig. 8c), predicted (filled) and measured (open) values paired by colour and joined by lines.
Nematic (T=1.05, R=2,3,4): groups director+amplitude (d=3), biaxial (d=2), all channels (d=5).  Measured: net ELBO / NLL gain of the v5 models (1.3 ablation without biaxial deviation; 1.4 full; their difference for biaxial).
Kagome: channel predictions (filled stars) and the measured VAE (ii)-(iv) value (open star, all channels together)."""
import numpy as np
from style import *
import fig_ll_core as L
from tables import num, nums
MP = {}
def fig14():
    D = L.D; c = D["cells"]; amp = D["amp"]; Rr = [2, 3, 4]; ncell = {2: 8, 3: 27, 4: 64}; nsite = {2: 216, 3: 64, 4: 27}; N = 1728
    T5 = L.T5; row = lambda R, pre, col: next(r[col] for r in T5 if int(num(r["R"])) == R and r["model"].startswith(pre))
    nll = {k: {R: num(row(R, k, "NLL gain vs R=1")) for R in Rr} for k in ("1.3", "1.4")}; net = {k: {R: nums(row(R, k, "net ELBO gain vs R=1"))[0] for R in Rr} for k in ("1.3", "1.4")}
    fig, ax = plt.subplots(figsize=(W2, 3.9)); G = np.logspace(-1.3, 2.0, 300)
    for dd, ls, yo in ((1, ":", 0.045), (2, "-", 0.0), (3, "--", -0.045)): ax.plot(G, L.univ(G, dd), ls, color=OI["grey"], lw=1.0); ax.text(120, L.univ(120, dd) + yo, f"$d={dd}$", fontsize=6.5, ha="left", va="center", color=OI["grey"])
    ax.axvspan(0.05, np.log(2), color=OI["verm"], alpha=0.10, lw=0); ax.text(0.06, 0.985, "$G\\leq\\ln 2$: gauge bit\n(criterion of the request;\nnot measured here)", fontsize=5.6, color=OI["verm"], va="top")
    groups = [("director + amplitude", OI["blue"], 3), ("biaxial", OI["green"], 2), ("all channels", OI["black"], 5)]; mk = {2: "o", 3: "s", 4: "^"}; first = True
    for R in Rr:
        g_dir = c["1.05"][str(R)]["G"]; g_amp = amp["1.05"][str(R)]["Ga"]; n = nsite[R]; gb = n * (nll["1.4"][R] - nll["1.3"][R]); gda = g_dir + g_amp                    # per-cell G of the groups
        f2 = lambda G_, d: float(L.univ(G_, d))
        pred = {"director + amplitude": (gda, (g_dir * f2(g_dir, 2) + g_amp * f2(g_amp, 1)) / gda), "biaxial": (gb, f2(gb, 2)), "all channels": (gda + gb, (g_dir * f2(g_dir, 2) + g_amp * f2(g_amp, 1) + gb * f2(gb, 2)) / (gda + gb))}
        gm_da = nll["1.3"][R] * N / ncell[R]; gm_all = nll["1.4"][R] * N / ncell[R]; gm_b = gm_all - gm_da
        meas = {"director + amplitude": (gm_da, net["1.3"][R] / (nll["1.3"][R] * N)), "biaxial": (gm_b, (net["1.4"][R] - net["1.3"][R]) / ((nll["1.4"][R] - nll["1.3"][R]) * N)), "all channels": (gm_all, net["1.4"][R] / (nll["1.4"][R] * N))}
        for name, col, dd in groups:
            (xp, yp), (xm, ym) = pred[name], meas[name]; ax.plot(xp, yp, mk[R], color=col, ms=5, label=None)
            if name == "biaxial": continue                                   # no separate measurement: difference of two models trained from different warm starts (a few nats), not used
            ax.plot([xp, xm], [yp, ym], "-", color=col, lw=0.8, alpha=0.8); ax.plot(xm, ym, mk[R], color=col, mfc="white", ms=5)
    kg = L.kag_channel_G()
    for key, d_, name in (("local", 3, "local frame"), ("ring", 1, "ring rotation"), ("defect", 1, "defect ($d{=}1$)")): ax.plot(kg[key], L.univ(kg[key], d_), "*", ms=10, color=COL_CH[key], mec="k", mew=0.3, label=f"{name} (pred.)")
    gm = 618.7 / 16; ax.plot(gm, 417.6 / 618.7, "*", ms=10, mfc="white", mec="k", label="VAE (ii)$-$(iv) (meas.)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color=col, ls="", ms=5) for _, col, _ in groups] + [Line2D([], [], marker="o", color="k", ls="", ms=5), Line2D([], [], marker="o", color="k", mfc="white", ls="", ms=5), Line2D([], [], marker="o", color="k", ls="", ms=5), Line2D([], [], marker="s", color="k", ls="", ms=5), Line2D([], [], marker="^", color="k", ls="", ms=5)]
    lab = ["director + amplitude", "biaxial (predicted only)", "all channels", "predicted (filled)", "measured, VAE v5 (open)", "$R=2$", "$R=3$", "$R=4$"]
    l1 = ax.legend(h, lab, frameon=False, fontsize=5.6, loc="lower right", ncol=2, bbox_to_anchor=(1.0, 0.0), title="nematic, $T=1.05$", title_fontsize=5.8, columnspacing=0.8)
    ax.add_artist(l1); hh, ll = ax.get_legend_handles_labels(); ax.legend(hh, ll, frameon=False, fontsize=5.6, loc="upper left", bbox_to_anchor=(0.0, 0.72), title="kagome (all channels: measured)", title_fontsize=5.6)
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.0); ax.set_xlim(0.05, 300); ax.set_xlabel("information gain per unit $G$ (nats; per cell for the nematic, per unit named in the legend for the kagome)"); ax.set_ylabel(r"net gain / $G$")
    savefig(fig, "fig14")
    MP["fig14"] = dict(title="Universal curve: prediction vs measurement (former Fig. 8c)", panels={"-": "ll_derived.json (G of director, amplitude), results/v4_table_A.md and v5_table_13_14.md (v5 models: 1.3 = no biaxial deviation, 1.4 = separated decoder + subspace; biaxial = 1.4 minus 1.3; T=1.05); kagome G: results/layered_gain*.txt, regress_k6u_T0.0020.json; kagome measured (ii)-(iv): REPORT_E3_v4 (net 417.6, NLL gain 618.7 nats per snapshot, 16 cells). Predicted points are on the curve by construction; measured (open) points are not."})
