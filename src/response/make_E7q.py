import sys
sys.path.insert(0, "/home/user/S2_VAE_for_spin_structures/figures")
from style import *
from ana_E7q import *
r = load_all(); names = {"Q": "amplitude m=0", "D": "director m=±1", "B": "biaxial m=±2"}; colR = {2: OI["blue"], 3: OI["green"], 4: OI["verm"]}
for T in (1.05, 0.95):
    rows = [("B", "model B (v5a)")] if T == 1.05 else []
    rows += [("A", "model A (v3h)")]; fig, ax = plt.subplots(len(rows), 3, figsize=(W2, 2.5 * len(rows) + 0.3), squeeze=False)
    for ri, (mod, lab) in enumerate(rows):
        for c, ch in enumerate("QDB"):
            a = ax[ri, c]; first = True
            for n, v in sorted(r.items()):
                if v["T"] != T or v["model"] != mod: continue
                k = np.sqrt(v["k2"]); sel = k > 0 if ch == "D" else k >= 0
                if first: a.plot(k[sel], v["mc"][c][sel], "o-", color="k", ms=2.5, lw=0.8, label="MC"); a.axhline(v["site"][c], color=OI["grey"], lw=0.6, ls=":", label="site"); first = False
                a.plot(k[sel], v["vae"][c][sel], "s-", color=colR[v["R"]], ms=2.5, lw=0.8, label=f"VAE R={v['R']}"); a.plot(k[sel], v["pred"][c][sel], "--", color=colR[v["R"]], lw=0.7)
                a.axvline(v["R"], color=colR[v["R"]], lw=0.4, ls="-.")
            a.set_yscale("log"); a.set_xlabel(r"$|k|$  ($q=2\pi|k|/12$)"); a.set_ylabel(r"$\chi(q)$"); a.set_title(f"{lab}, {names[ch]}", fontsize=7)
            if ri == 0 and c == 0: a.legend(frameon=False, fontsize=5.5, loc="upper right")
    for i, a in enumerate(ax.ravel()): panel(a, "abcdef"[i])
    fig.tight_layout(); savefig(fig, f"E7_fig_chiq_T{T}")
