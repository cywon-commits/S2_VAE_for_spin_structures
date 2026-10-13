"""Figures + criteria table from results/{oracle,eval,ablate_*}.json.  python3 make_report.py"""
import json, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "../results"; orc = json.load(open(f"{R}/oracle.json")); ev = json.load(open(f"{R}/eval.json"))
abl = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/ablate/*.json"))]
key = lambda r: (r["T"], r["st"])
def sel(**kw): return sorted([r for r in ev if all(r[k] == v for k, v in kw.items())], key=key)
g1 = sel(R=1, beta=1.0)
# --- fig 1: a2, a4 vs T (trained R=1 vs oracle)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for k, (nm, name) in enumerate([("a2", "$a_2$"), ("a4", "$a_4$")]):
    ax[k].plot([r["T"] for r in orc], [r[nm] for r in orc], "ko-", label="oracle")
    for R_, mk in [(1, "s--"), (2, "^:"), (4, "v:")]:
        g = sel(R=R_, beta=1.0)
        if g: ax[k].plot([r["T"] for r in g], [r[nm] for r in g], mk, label=f"VAE R={R_}")
    ax[k].set_xlabel("T/J"); ax[k].set_ylabel(name); ax[k].legend(fontsize=7)
ax[1].axhline(0, c="gray", lw=.5)
ax[2].plot([r["T"] for r in orc], [r["a2_MF"] for r in orc], "k--", label="mean field 6S/T")
ax[2].plot([r["T"] for r in orc], [r["a2"] for r in orc], "ko-", label="oracle $a_2$")
for R_, mk in [(1, "s--"), (2, "^:"), (4, "v:")]:
    g = sel(R=R_, beta=1.0)
    if g: ax[2].plot([r["T"] for r in g], [r["a2"] for r in g], mk, label=f"VAE R={R_}")
ax[2].set_xlabel("T/J"); ax[2].set_ylabel("$a_2$"); ax[2].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{R}/fig1_a2_a4_vs_T.png", dpi=130); plt.close()
# --- fig 2: sigma^2 vs R per T
fig, ax = plt.subplots(figsize=(5, 3.6))
for T in sorted({r["T"] for r in ev if r["st"] == "o" and r["T"] < 1.13}):
    rs = sorted([r for r in ev if r["T"] == T and r["st"] == "o" and r["beta"] == 1.0 and r["sigma2"] is not None], key=lambda r: r["R"])
    o = [r for r in orc if r["T"] == T and r["st"] == "o"]
    if rs: ax.plot([r["R"] for r in rs], [r["sigma2"] for r in rs], "o-", label=f"T={T}")
ax.set_xlabel("latent resolution R (1 = global)"); ax.set_ylabel(r"$\sigma^2=-\frac{35}{18}a_4/a_2^2$"); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{R}/fig2_sigma2_vs_R.png", dpi=130); plt.close()
# --- fig 3: latent power per l vs T, and KL
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for l in range(4): ax[0].plot([r["T"] for r in g1], [r["latent_power_frac"][l] for r in g1], "o-", label=f"l={l}")
ax[0].set_xlabel("T/J"); ax[0].set_ylabel("latent power fraction"); ax[0].legend(fontsize=7)
ax[1].plot([r["T"] for r in g1], [r["kl"] for r in g1], "ko-"); ax[1].set_xlabel("T/J"); ax[1].set_ylabel("KL per configuration (nats)")
plt.tight_layout(); plt.savefig(f"{R}/fig3_latent_power_vs_T.png", dpi=130); plt.close()
# --- fig 4: dNLL vs L
if abl:
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for r in sorted(abl, key=lambda r: r["T"]):
        ax.plot([1, 2, 4, 6], [r["dNLL"]["1"], r["dNLL"]["2"], r["dNLL"]["4"], 0], "o-", label=f"VAE T={r['T']}")
    for o in orc:
        if o["st"] == "o" and any(abs(o["T"] - r["T"]) < 1e-6 for r in abl): ax.plot([1, 2, 4, 6], [o["dNLL"]["1"], o["dNLL"]["2"], o["dNLL"]["4"], 0], "k:", lw=.8)
    ax.set_yscale("symlog", linthresh=1e-3); ax.set_xlabel("decoder cutoff L"); ax.set_ylabel(r"$\Delta$NLL vs L=6 (nats/site)"); ax.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(f"{R}/fig4_dNLL_vs_L.png", dpi=130); plt.close()
# --- tables
L = []
L.append("| T | st | S | a2 VAE | a2 oracle | a4 VAE | a4 oracle | odd/even VAE | odd/even oracle | KL | l=1 latent frac | l=2 latent frac | sigma2 VAE | sigma2 oracle |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in g1:
    o = [x for x in orc if key(x) == key(r)][0]; f = lambda v, p=3: "-" if v is None else f"{v:.{p}g}"
    L.append(f"| {r['T']} | {r['st']} | {r['S']:.3f} | {r['a2']:.3f} | {o['a2']:.3f} | {r['a4']:+.3f} | {o['a4']:+.3f} | {r['odd_even']:.1e} | {o['odd_even']:.1e} | {r['kl']:.0f} | {r['latent_power_frac'][1]:.3f} | {r['latent_power_frac'][2]:.3f} | {f(r['sigma2'])} | {f(o['sigma2'])} |")
open(f"{R}/table_E1.md", "w").write("\n".join(L) + "\n")
L = ["| T | R | a2 | a4 | sigma2 | a2 MF 6S/T | KL |\n|---|---|---|---|---|---|---|"]
for r in sorted([r for r in ev if r["st"] == "o" and r["beta"] == 1.0], key=lambda r: (r["T"], r["R"])):
    o = [x for x in orc if key(x) == key(r)][0]
    L.append(f"| {r['T']} | {r['R']} | {r['a2']:.3f} | {r['a4']:+.3f} | {'-' if r['sigma2'] is None else format(r['sigma2'], '.3f')} | {o['a2_MF']:.3f} | {r['kl']:.0f} |")
open(f"{R}/table_E2.md", "w").write("\n".join(L) + "\n")
L = ["| T | R | beta | NLL(L=6)/site | KL | a2 | a4 | odd/even | l=1 frac |\n|---|---|---|---|---|---|---|---|---|"]
for r in sorted([r for r in ev if r["R"] == 1 and r["T"] in (0.95, 1.1) and r["st"] == "o"], key=lambda r: (r["T"], r["beta"])):
    L.append(f"| {r['T']} | 1 | {r['beta']} | {r['nll']['6']:.4f} | {r['kl']:.0f} | {r['a2']:.3f} | {r['a4']:+.3f} | {r['odd_even']:.1e} | {r['latent_power_frac'][1]:.3f} |")
open(f"{R}/table_beta.md", "w").write("\n".join(L) + "\n")
L = ["| T | L=1 VAE | L=1 oracle | L=2 VAE | L=2 oracle | L=4 VAE | L=4 oracle |\n|---|---|---|---|---|---|---|"]
for r in sorted(abl, key=lambda r: r["T"]):
    o = [x for x in orc if key(x) == key(r)][0]; d = r["dNLL"]
    L.append(f"| {r['T']} | {d['1']:.3f} | {o['dNLL']['1']:.3f} | {d['2']:.4f} | {o['dNLL']['2']:.4f} | {d['4']:.4f} | {o['dNLL']['4']:.4f} |")
open(f"{R}/table_f.md", "w").write("\n".join(L) + "\n")
L = ["| T | R | time-reversal rel. change per latent l=0,1,2,3 |\n|---|---|---|"]
for r in g1: L.append(f"| {r['T']}{r['st']} | 1 | " + ", ".join(f"{x:.2g}" for x in r["time_reversal_rel"]) + " |")
open(f"{R}/table_e.md", "w").write("\n".join(L) + "\n")
print("ok")
