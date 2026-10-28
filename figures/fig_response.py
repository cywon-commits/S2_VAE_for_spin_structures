"""Fig. 15: quadrupolar susceptibility of the nematic VAEs (aggregated posterior), decoder + latent parts, against the MC fluctuation value."""
import numpy as np, matplotlib.pyplot as plt
from style import W1, OI, COL_LL, COL_CH, savefig, load, panel
D = load("results/response/A_nem.json")
groups = [("v3h", 0.95), ("v3h", 1.05), ("v5a", 1.05)]
labels = {("v3h", 0.95): "model A\n$T=0.95J$", ("v3h", 1.05): "model A\n$T=1.05J$", ("v5a", 1.05): "model B\n$T=1.05J$"}
fig, ax = plt.subplots(figsize=(W1, 2.45))
x0 = 0; xt = []
for g in groups:
    rows = sorted([d for d in D if (d["tag"], d["T"]) == g], key=lambda d: d["R"])
    xs = x0 + np.arange(len(rows)) * 0.8
    dec = np.array([r["posterior"]["quad_dec"] for r in rows]); lat = np.array([r["posterior"]["quad_lat"] for r in rows])
    lse = np.array([r["posterior"]["quad_lat_se"] for r in rows])
    ax.bar(xs, dec, 0.62, color=OI["grey"], alpha=0.55, label="decoder $\\chi_{\\rm dec}$" if x0 == 0 else None)
    ax.bar(xs, lat, 0.62, bottom=dec, color=COL_CH["amp"], label="latent $\\chi_{\\rm lat}$" if x0 == 0 else None)
    ax.errorbar(xs, dec + lat, lse, fmt="none", ecolor="k", elinewidth=0.7, capsize=1.5)
    f, fs = rows[0]["quad_fdt"], rows[0]["quad_fdt_se"]
    ax.fill_between([xs[0] - 0.45, xs[-1] + 0.45], f - fs, f + fs, color="k", alpha=0.15, lw=0)
    ax.plot([xs[0] - 0.45, xs[-1] + 0.45], [f, f], "k-", lw=1.0, label="MC $\\chi_{\\rm FDT}$" if x0 == 0 else None)
    ax.plot([xs[0] - 0.45, xs[-1] + 0.45], [rows[0]["quad_weiss"]] * 2, ls=":", color=OI["blue"], lw=1.0, label="frozen Weiss $\\chi_W$" if x0 == 0 else None)
    for xx, r in zip(xs, rows): ax.text(xx, -0.05, f"{r['R']}", ha="center", va="top", fontsize=6.5)
    xt.append((xs.mean(), labels[g])); x0 = xs[-1] + 1.4
ax.set_xticks([p for p, _ in xt]); ax.set_xticklabels([l for _, l in xt], fontsize=7)
ax.tick_params(axis="x", length=0, pad=11)
ax.text(-0.75, -0.05, "$R$", ha="right", va="top", fontsize=6.5)
ax.set_ylim(0, 2.25); ax.set_xlim(-0.8, x0 - 0.9)
ax.set_ylabel("quadrupolar susceptibility $\\chi_Q$ (per site)")
ax.legend(frameon=False, loc="upper right", fontsize=6.5, ncol=2, columnspacing=0.8, handlelength=1.4)
savefig(fig, "fig15")
