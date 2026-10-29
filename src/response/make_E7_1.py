import sys, json, glob, os
sys.path.insert(0, "/home/user/S2_VAE_for_spin_structures/figures")
from style import *
from ana_C6 import load as loadC
RS = f"{ROOT}/results/response"; d = json.load(open(f"{RS}/E7_1ab.json"))
# ---- Fig: S distributions
sel = [n for n in ("v5a_T1.05o_R2_b1", "v5a_T1.05o_R4_b1", "v3h_T1.05o_R2_b1", "v3h_T0.95o_R2_b1") if n in d]; fig, ax = plt.subplots(1, len(sel), figsize=(W2, 2.2), squeeze=False)
for a, n in zip(ax[0], sel):
    r = d[n]; bins = np.linspace(0, 0.8, 60)
    a.hist(r["S_data_samples"], bins, density=True, color=OI["black"], alpha=0.5, label="data"); a.hist(r["S_old_samples"], bins, density=True, histtype="step", color=OI["verm"], label="old prior"); a.hist(r["S_new_samples"], bins, density=True, histtype="step", color=OI["blue"], lw=1.2, label="refit prior")
    a.set_xlabel("S"); a.set_title(f"{'B' if 'v5a' in n else 'A'}: T={r['T']}, R={r['R']}", fontsize=7)
ax[0, 0].set_ylabel("density"); ax[0, 0].legend(frameon=False, fontsize=5.5)
for i, a in enumerate(ax[0]): panel(a, "abcd"[i])
fig.tight_layout(); savefig(fig, "E7_fig_Sdist")
# ---- Fig: nonlinear
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.6)); C = loadC(1.05); Ds = np.array(sorted(C)); lab = ["S", r"$\langle P_2(d\cdot e)\rangle$", r"$\langle B\rangle$"]
for k, key in enumerate("SAB"):
    ax[k].errorbar(Ds, [C[D][key][0] for D in Ds], [C[D][key][1] for D in Ds], fmt="o", ms=3, color="k", label="MC")
    for f in sorted(g for g in glob.glob(f"{RS}/E7_C_*.json") if "k200" not in g):
        e = json.load(open(f)); mod = "B" if "v5a" in f else "A"; col = OI["verm"] if mod == "B" else OI["blue"]; ls = "-" if e["R"] == 2 else "--"; h = sorted(e["hmc"], key=lambda r: r["D"])
        ax[k].plot([r["D"] for r in h], [r[key] for r in h], ls, color=col, marker="x", ms=4, label=f"refit {mod} R={e['R']}")
    for f in sorted(glob.glob(f"{RS}/E6_B_*T1.05o_R2_b1.json")):
        e = json.load(open(f)); mod = "B" if "v5a" in f else "A"; col = OI["verm"] if mod == "B" else OI["blue"]; h = sorted(e["hmc"], key=lambda r: r["D"])
        ax[k].plot([r["D"] for r in h], [r[key] for r in h], ":", color=col, lw=0.9, label=f"old {mod} R=2")
    ax[k].set_xlabel("D"); ax[k].set_ylabel(lab[k]); ax[k].set_xticks([-0.05, 0, 0.05])
h_, l_ = ax[0].get_handles_labels() if False else ax[0].get_legend_handles_labels(); seen = {}; [seen.setdefault(l, h) for h, l in zip(h_, l_)]
fig.subplots_adjust(wspace=0.5); fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=4, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.2))
for i, a in enumerate(ax): panel(a, "abc"[i])
savefig(fig, "E7_fig_nonlinear_refit")
