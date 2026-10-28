"""E5 figures + tables.  python3 make_E5.py"""
import json, os, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ana_B import ana
ROOT = "/home/user/S2_VAE_for_spin_structures"; RS = f"{ROOT}/results/response"; HE = f"{ROOT}/src/response"; os.makedirs(f"{ROOT}/figures/E5", exist_ok=True)
scan = json.load(open(f"{HE}/out/scan_summary.json"))
# ---------------- Fig 1: kagome m(h)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
for T, c in ((0.001, "k"), (0.005, "0.5")):
    S = [s for s in scan if s["T"] == T]; ax[0].plot([s["h"] for s in S], [s["m"] for s in S], "o", ms=3.5, color=c, label=f"MC T={T}")
hh = np.logspace(-6, np.log10(6.5), 300); ax[0].plot(hh, np.minimum(hh / 6, 1), "k--", lw=0.7, label="T=0 / MC: m=h/6")
LBL = lambda n: {"kag": "R=1", "kagu": "R="+n.split("_R")[-1], "kag6": "R="+n.split("_R")[-1]+" (kag6)"}[n.split("_")[0]]
cols = {"kag_T0.0005_R1": "C0", "kagu_T0.0005_R4": "C1", "kag6_T0.0005_R4": "C2", "kagu_T0.0005_R12": "C3"}
for f in sorted(glob.glob(f"{RS}/B_kag*_T0.0005_*.npz")):
    name = os.path.basename(f)[2:-4]; r = ana(f); h = np.array([x["h"] for x in r]); m = np.array([x["m_plot"] for x in r]); e = np.array([x["ess"] for x in r]); c = cols.get(name, "C4")
    ok = e >= 100; ax[0].plot(h[ok], m[ok], "-o", ms=3, color=c, label=f"VAE reweight {LBL(name)}"); ax[0].plot(h[~ok], m[~ok], ":", color=c, lw=1)
    ax[1].plot(h[ok], m[ok], "-o", ms=3, color=c)
    ax[1].plot(h[~ok], m[~ok], ":", color=c, lw=1)
for f in sorted(glob.glob(f"{RS}/Bmc_kag*_T0.0005_*.json")):
    name = os.path.basename(f)[4:-5]; d = json.load(open(f)); c = cols.get(name, "C4")
    ax[0].plot([r["h"] for r in d], [r["m"] for r in d], "x", ms=6, color=c, label=f"VAE HMC {LBL(name)}")
for S_ in [s_ for s_ in scan if s_["T"] == 0.001]:
    pass
ax[0].axvspan(1e-6, 1.5e-5, color="C2", alpha=0.08)
ax[0].set_xscale("log"); ax[0].set_xlim(1e-6, 7); ax[0].set_ylim(-0.02, 1.05); ax[0].set_xlabel("h / J"); ax[0].set_ylabel("m"); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("solid: ESS>=100, dotted: ESS<100", fontsize=8)
hh = np.logspace(-6, -1, 50); ax[1].plot(hh, hh / 6, "k--", lw=0.7, label="MC: chi = 1/6"); ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlim(1e-6, 1e-1); ax[1].set_ylim(1e-7, 1.5); ax[1].set_xlabel("h / J"); ax[1].set_ylabel("m (linear region)"); ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{ROOT}/figures/E5/fig1_kagome_mh.png", dpi=150)
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for T, c in ((0.001, "k"), (0.005, "0.5")):
    S = [s_ for s_ in scan if s_["T"] == T]; ax.plot([s_["h"] for s_ in S], [s_["nu_h2"] for s_ in S], "o-", ms=3, color=c, label=f"MC T={T}")
for f in sorted(glob.glob(f"{RS}/B_kag*_T0.0005_*.npz")):
    name = os.path.basename(f)[2:-4]; r = ana(f); c = cols.get(name, "C4"); h = np.array([x["h"] for x in r]); v = np.array([x["nuh2"] for x in r]); e = np.array([x["ess"] for x in r])
    ax.plot(h[e >= 100], v[e >= 100], "-o", ms=3, color=c, label=f"VAE {LBL(name)} (ESS>=100)")
for f in sorted(glob.glob(f"{RS}/Bmc_kag*_T0.0005_*.json")):
    name = os.path.basename(f)[4:-5]; d = json.load(open(f)); ax.plot([r["h"] for r in d], [r["nuh2"] for r in d], "x", color=cols.get(name, "C4"), label=f"VAE HMC {LBL(name)}")
ax.axhline(1 / 3, color="gray", lw=0.6, ls=":"); ax.set_xscale("log"); ax.set_xlabel("h / J"); ax.set_ylabel(r"$\langle(\hat\nu\cdot\hat h)^2\rangle$"); ax.legend(frameon=False, fontsize=6); fig.tight_layout(); fig.savefig(f"{ROOT}/figures/E5/fig3_nu_h2.png", dpi=150)
# ---------------- Fig 2: chi_VAE / chi_FDT
A = json.load(open(f"{RS}/A_kag.json")); N = json.load(open(f"{RS}/A_nem.json"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for T, mk in ((0.0005, "o"), (0.002, "s")):
    for kind, ls in (("posterior", "-"),):
        pts = [(r["R"], (r[kind]["dec_in"] + r[kind]["lat_in"]) / r["fdt_in"]) for r in A if r["T"] == T and r["run"].startswith("kagu") or (r["T"] == T and r["run"].startswith("kag_T") and "_p" not in r["run"])]
        pts = sorted(set(pts)); ax[0].plot([p[0] for p in pts], [p[1] for p in pts], mk + ls, label=f"T={T} (agg. posterior)")
    pts = sorted(set((r["R"], (r["prior"]["dec_in"] + r["prior"]["lat_in"]) / r["fdt_in"]) for r in A if r["T"] == T and "prior" in r and (r["run"].startswith("kagu") or r["run"].startswith("kag_T")))); ax[0].plot([p[0] for p in pts], [p[1] for p in pts], mk + ":", label=f"T={T} (prior)")
ax[0].axhline(1, color="k", lw=0.6); ax[0].set_yscale("log"); ax[0].set_xlabel("R"); ax[0].set_ylabel(r"$\chi_{VAE,\parallel}/\chi_{FDT}$ (in-plane)"); ax[0].legend(frameon=False, fontsize=7)
for tag, mk in (("v3h", "o"), ("v5a", "s")):
    for kind, ls in (("posterior", "-"), ("prior", ":")):
        pts = sorted((r["T"] + 0.003 * r["R"] / 4, (r[kind]["quad_dec"] + r[kind]["quad_lat"]) / r["quad_fdt"]) for r in N if r["tag"] == tag)
        if pts: ax[1].plot([p[0] for p in pts], [p[1] for p in pts], mk, ls=ls, label=f"{tag} {kind}")
ax[1].axhline(1, color="k", lw=0.6); ax[1].set_yscale("log"); ax[1].set_xlabel("T"); ax[1].set_ylabel(r"$\chi^{quad}_{VAE}/\chi^{quad}_{FDT}$"); ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{ROOT}/figures/E5/fig2_ratio.png", dpi=150)
