"""Fig. 15: response of the nematic VAEs to multipolar fields (aggregated posterior), decoder + latent parts, against the MC fluctuation value.
(a) amplitude channel m=0, quadrupolar field P2(n.d) [results/response/A_nem.json]; (b) biaxial channel m=+-2 [results/response/E6_A.json, MC: src/response/fdt_biax.json]."""
import numpy as np, matplotlib.pyplot as plt
from style import W1, OI, COL_CH, savefig, load, panel
A = load("results/response/A_nem.json"); E = load("results/response/E6_A.json"); FB = load("src/response/fdt_biax.json")
groups = [("v3h", "A", 0.95), ("v3h", "A", 1.05), ("v5a", "B", 1.05)]
fig, axs = plt.subplots(2, 1, figsize=(W1, 4.3), sharex=True)
def draw(ax, rows_fn, ref_fn, site_fn, ymax, col, first_label):
    x0 = 0; xt = []
    for tag, mod, T in groups:
        rows = sorted(rows_fn(tag, T), key=lambda r: r[0])
        xs = x0 + np.arange(len(rows)) * 0.8
        dec = np.array([r[1] for r in rows]); lat = np.array([r[2] for r in rows]); se = np.array([r[3] for r in rows])
        ax.bar(xs, dec, 0.62, color=OI["grey"], alpha=0.55, label="decoder $\\chi_{\\rm dec}$" if x0 == 0 else None)
        ax.bar(xs, lat, 0.62, bottom=dec, color=col, label=first_label if x0 == 0 else None)
        ax.errorbar(xs, dec + lat, se, fmt="none", ecolor="k", elinewidth=0.7, capsize=1.5)
        f, fs = ref_fn(tag, T); lo, hi = xs[0] - 0.45, xs[-1] + 0.45
        ax.fill_between([lo, hi], f - fs, f + fs, color="k", alpha=0.15, lw=0)
        ax.plot([lo, hi], [f, f], "k-", lw=1.0, label="MC $\\chi_{\\rm FDT}$" if x0 == 0 else None)
        ax.plot([lo, hi], [site_fn(T)] * 2, ls=":", color=OI["blue"], lw=1.0, label="single site" if x0 == 0 else None)
        for xx, r in zip(xs, rows): ax.text(xx, -0.035 * ymax, f"{r[0]}", ha="center", va="top", fontsize=6.5)
        xt.append((xs.mean(), f"model {mod}\n$T={T:.2f}J$")); x0 = xs[-1] + 1.4
    ax.set_ylim(0, ymax); ax.set_xlim(-0.8, x0 - 0.9)
    return xt
def qrows(tag, T): return [(d["R"], d["posterior"]["quad_dec"], d["posterior"]["quad_lat"], d["posterior"]["quad_lat_se"]) for d in A if (d["tag"], d["T"]) == (tag, T)]
def qref(tag, T): d = [d for d in A if (d["tag"], d["T"]) == (tag, T)][0]; return d["quad_fdt"], d["quad_fdt_se"]
def qsite(T): return [d for d in A if d["T"] == T][0]["quad_weiss"]
xt = draw(axs[0], qrows, qref, qsite, 2.25, COL_CH["amp"], "latent $\\chi_{\\rm lat}$")
axs[0].set_ylabel("amplitude $\\chi_Q$ ($m=0$)")
axs[0].legend(frameon=False, loc="upper right", fontsize=6.5, ncol=2, columnspacing=0.8, handlelength=1.4)
tagmod = {"v3h": "A", "v5a": "B"}
def brows(tag, T): return [(d["R"], d["resp"]["B_dec"], d["resp"]["B_lat"], d["resp"]["B_lat_se"]) for d in E if (d["model"], d["T"]) == (tagmod[tag], T)]
def bref(tag, T): f = FB[str(T)]; return f["chi_B"], f["chi_B_se"]
def bsite(T): return FB[str(T)]["chi_B_site"]
draw(axs[1], brows, bref, bsite, 0.42, COL_CH["biax"], "latent $\\chi_{\\rm lat}$")
axs[1].set_ylabel("biaxial $\\chi_B$ ($m=\\pm2$)")
axs[1].legend(frameon=False, loc="upper left", fontsize=6.5, ncol=2, columnspacing=0.8, handlelength=1.4)
axs[1].set_xticks([p for p, _ in xt]); axs[1].set_xticklabels([l for _, l in xt], fontsize=7)
for ax in axs: ax.tick_params(axis="x", length=0, pad=11)
panel(axs[0], "(a)"); panel(axs[1], "(b)")
fig.tight_layout(h_pad=1.2)
savefig(fig, "fig15")
