import json, glob, os, sys, numpy as np
sys.path.insert(0, "/home/user/S2_VAE_for_spin_structures/figures")
from style import *
from ana_C6 import load as loadC
RS = f"{ROOT}/results/response"; A = json.load(open(f"{RS}/E6_A.json"))
# ---------- Fig 1: channel-resolved chi_VAE/chi_FDT
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.8), sharey=True)
Qf = {0.95: 0.39, 1.05: 0.90}; Bf = {0.95: 0.129, 1.05: 0.212}
for k, (T, mod, ttl) in enumerate([(0.95, "A", "T=0.95, model A"), (1.05, "A", "T=1.05, model A"), (1.05, "B", "T=1.05, model B")]):
    rows = [r for r in A if r["T"] == T and r["model"] == mod]; Rs = [r["R"] for r in rows]; x = np.arange(len(Rs)) + (k * 4)
    qv = [(r["resp"]["Q_dec"] + r["resp"]["Q_lat"]) / Qf[T] for r in rows]; bv = [(r["resp"]["B_dec"] + r["resp"]["B_lat"]) / Bf[T] for r in rows]
    ax[0].bar(x - 0.2, qv, 0.4, color=OI["orange"], label="m=0 (amplitude)" if k == 0 else None); ax[0].bar(x + 0.2, bv, 0.4, color=OI["green"], label="m=±2 (biaxial)" if k == 0 else None)
    ax[0].set_xticks(list(ax[0].get_xticks()) + list(x)); 
    for xi, R in zip(x, Rs): ax[0].text(xi, 0.03, f"R={R}", ha="center", fontsize=6, rotation=90, va="bottom")
    ax[1].bar(x - 0.2, [r["resp"]["Q_dec"] / Qf[T] for r in rows], 0.4, color=OI["orange"], alpha=0.5, label="decoder part" if k == 0 else None); ax[1].bar(x + 0.2, [r["resp"]["B_dec"] / Bf[T] for r in rows], 0.4, color=OI["green"], alpha=0.5)
    ax[0].text(x.mean(), 3.05, ttl, ha="center", fontsize=7)
for a in ax: a.axhline(1, color="k", lw=0.6); a.set_xticks([]); a.set_ylim(0, 3.3)
ax[0].set_ylabel(r"$\chi_{VAE}/\chi_{FDT}$"); ax[0].legend(frameon=False, loc="upper right", fontsize=6); ax[1].set_title("decoder part only", fontsize=7); ax[0].set_title("total (agg. posterior)", fontsize=7)
panel(ax[0], "a"); panel(ax[1], "b"); savefig(fig, "E6_fig1_channels")
# ---------- Fig 2: S(D), A(D), B(D), T=1.05
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.6)); C = loadC(1.05); Ds = np.array(sorted(C)); mcol = OI["black"]
for k, key in enumerate("SAB"):
    ax[k].errorbar(Ds, [C[D][key][0] for D in Ds], [C[D][key][1] for D in Ds], fmt="o", ms=3, color=mcol, label="MC")
    for f, c, ls in sorted((f, c, l) for f, c, l in [(g, OI["blue"] if "v3h" in g else OI["verm"], "-" if "R4" in g else "--") for g in glob.glob(f"{RS}/E6_B_*T1.05*.json")]):
        d = json.load(open(f)); h = sorted(d["hmc"], key=lambda r: r["D"]); ax[k].plot([r["D"] for r in h], [r[key] for r in h], ls, color=c, marker="x", ms=4, label=("model A" if "v3h" in f else "model B") + f" R={d['R']} HMC")
    ax[k].set_xlabel("D"); ax[k].set_ylabel([r"$S$", r"$\langle P_2(d\cdot e)\rangle$", r"$\langle B\rangle$"][k])
Dl = np.linspace(-0.05, 0.05, 50); ax[0].plot(Dl, C[0][ "S"][0] + 0.90 * np.where(Dl > 0, Dl, -Dl / 2) * 0 + 0.90 * np.where(Dl > 0, Dl, np.abs(Dl) / 2), ":", color=OI["grey"], label=r"linear: $\chi_Q\,\epsilon$ (FDT)")
ax[0].legend(frameon=False, fontsize=5.5); panel(ax[0], "a"); panel(ax[1], "b"); panel(ax[2], "c"); savefig(fig, "E6_fig2_nonlinear")
