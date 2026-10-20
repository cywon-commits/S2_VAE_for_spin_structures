"""Figure 14: both systems on one universal-curve axis.  Same data sources as Fig. 8c and Fig. 13 (see manifest)."""
import numpy as np
from style import *
import fig_ll_core as L
from tables import num, nums
MP = {}
def fig14():
    D = L.D; c = D["cells"]; amp = D["amp"]; Rr = [2, 3, 4]; nsite = {2: 216, 3: 64, 4: 27}; biax = {2: 0.004, 3: 0.013, 4: 0.025}
    fig, ax = plt.subplots(figsize=(W1 * 1.15, 3.6)); G = np.logspace(-1.3, 2.0, 300)
    for dd, ls in ((1, ":"), (2, "-"), (3, "--")): ax.plot(G, L.univ(G, dd), ls, color=OI["grey"], lw=1.0); ax.text(80, L.univ(80, dd) + (0.035 if dd == 1 else -0.035 if dd == 3 else 0.0), f"$d={dd}$", fontsize=6, ha="right", va="center", color=OI["grey"])
    ax.axvspan(0.05, np.log(2), color=OI["verm"], alpha=0.12, lw=0); ax.text(0.085, 0.93, r"$G\leq\ln 2$" + "\ngauge bit\n(criterion of the request;\nnot measured here)", fontsize=5.5, va="top", color=OI["verm"])
    for R, mk in zip(Rr, ("o", "s", "^")):
        Gd = c["1.05"][str(R)]["G"]; ax.plot(Gd, L.univ(Gd, 2), mk, color=COL_CH["director"], label="nematic director" if R == 2 else None); Ga = amp["1.05"][str(R)]["Ga"]; ax.plot(Ga, L.univ(Ga, 1), mk, color=COL_CH["amp"], label="nematic amplitude" if R == 2 else None)
        gb = nsite[R] * biax[R]; ax.plot(gb, L.univ(gb, 2), mk, color=COL_CH["biax"], label="nematic biaxial" if R == 2 else None)
    kg = L.kag_channel_G()
    for key, d_, name in (("local", 3, "kagome local frame"), ("ring", 1, "kagome ring rotation"), ("defect", 1, "kagome defect ($d\\!=\\!1$ assumed)")): ax.plot(kg[key], L.univ(kg[key], d_), "*", ms=9, color=COL_CH[key], mec="k", mew=0.3, label=name)
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.0); ax.set_xlim(0.05, 100); ax.set_xlabel("information gain per unit $G$ (nats)"); ax.set_ylabel(r"predicted net gain / $G=1-\frac{d}{2G}\ln(1+\frac{2G}{d})$")
    ax.legend(frameon=False, fontsize=5.8, loc="lower right"); savefig(fig, "fig14")
    MP["fig14"] = dict(title="Two systems on one curve", panels={"-": "G values: figures/data/ll_derived.json (nematic, T=1.05, R=2,3,4), results/layered_gain*.txt and regress_k6u_T0.0020.json (kagome); curve: analytic. Points sit on the curve by construction (x measured, y predicted); measured VAE net gains are shown in Fig. 8c"})
