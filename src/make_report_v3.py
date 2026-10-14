"""v3 tables and figures.  python3 make_report_v3.py"""
import json, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R_ = "../results"; N = 1728
orc = {(r["T"], r["st"]): r for r in json.load(open(f"{R_}/oracle.json"))}; MF = {T: r["a2_MF"] for (T, st), r in orc.items() if st == "o"}
ALOC = {0.80: 5.77, 0.95: 4.22, 1.05: 3.13, 1.10: 2.34}; S2SITE = {0.80: 0.046, 0.95: 0.062, 1.05: 0.068, 1.10: 0.073}
PS2 = {0.80: [0.046, 0.040, 0.031, 0.024], 0.95: [0.062, 0.052, 0.040, 0.028], 1.05: [0.068, 0.055, 0.035, 0.015]}
PNLL = {0.80: [.042, .099, .150], 0.95: [.043, .094, .144], 1.05: [.034, .088, .139], 1.10: [.040, .098, .160]}
PNET = {0.80: [55, 117, 155], 0.95: [56, 110, 148], 1.05: [42, 101, 140], 1.10: [50, 116, 169]}; PKL = [19, 53, 102]
mom = {}
for f in glob.glob(f"{R_}/v3/moments_*.json"):
    for r in json.load(open(f)): mom[(r["T"], r["R"])] = r
base = {}; ind = {}
for f in glob.glob(f"{R_}/v2p/evalA_*R1*.json"):
    r = json.load(open(f)); base[r["T"]] = r
for f in glob.glob(f"{R_}/v2B/evalA_*.json"):
    r = json.load(open(f)); ind[(r["T"], r["R"])] = r
hier = {}
for f in glob.glob(f"{R_}/v3h/eval_h*.json"):
    for r in json.load(open(f)): hier[(r["T"], r["R"])] = r
Ts = sorted({T for (T, R) in mom if R > 1})
# ---------------- 2.1
L = ["| T | R | a_loc (per-sample est) | a_loc (mean-z) | MF 6S/T | data a_loc | sigma2 (per-sample est) | sigma2 (avg moments) | sigma2 (mean-z) | predicted sigma2 | frac. negative |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for T in sorted({T for (T, R) in mom}):
    for R in (1, 2, 3, 4):
        if (T, R) not in mom: continue
        r = mom[(T, R)]; ps = PS2.get(T, [None] * 4)[R - 1]
        L.append(f"| {T} | {R} | {r['samp_each']['a_loc']:.3f} | {r['mean']['a_loc']:.3f} | {MF[T]:.3f} | {ALOC.get(T, float('nan')):.2f} | {r['samp_each']['sigma2']:.4f} | {r['samp']['sigma2']:.4f} | {r['mean']['sigma2']:.4f} | {'-' if ps is None else ps} | {r['samp_each']['frac_neg']:.2f} |")
open(f"{R_}/v3_table_2p1.md", "w").write("\n".join(L) + "\n")
col = {0.80: "C0", 0.95: "C1", 1.05: "C2", 1.10: "C3"}
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for T in Ts:
    Rs = [R for R in (1, 2, 3, 4) if (T, R) in mom]
    ax[0].plot(Rs, [mom[(T, R)]["samp_each"]["sigma2"] for R in Rs], "o-", c=col[T], label=f"T={T}"); ax[0].plot(Rs, [mom[(T, R)]["mean"]["sigma2"] for R in Rs], "s:", c=col[T], alpha=.6)
    if T in PS2: ax[0].plot([1, 2, 3, 4], PS2[T], "x--", c=col[T], alpha=.6)
    ax[1].plot(Rs, [mom[(T, R)]["samp_each"]["a_loc"] for R in Rs], "o-", c=col[T], label=f"T={T}"); ax[1].axhline(MF[T], c=col[T], ls=":", lw=1)
    ax[2].plot(Rs, [mom[(T, R)]["mean"]["a_loc"] for R in Rs], "s-", c=col[T], label=f"T={T}"); ax[2].axhline(MF[T], c=col[T], ls=":", lw=1)
ax[0].set_ylabel(r"$\sigma^2$ (moment estimator)"); ax[1].set_ylabel(r"$a_{loc}$ (per-sample estimate; dotted: $6S/T$)"); ax[2].set_ylabel(r"$a_{loc}$ (mean-z decode)")
for a in ax: a.set_xlabel("latent resolution R"); a.set_xticks([1, 2, 3, 4])
ax[0].legend(fontsize=6); ax[0].text(.02, .03, "solid: per-sample est, s:: mean-z, x--: chat prediction", transform=ax[0].transAxes, fontsize=6)
plt.tight_layout(); plt.savefig(f"{R_}/v3_fig_2p1.png", dpi=130); plt.close()
# ---------------- 2.2
if hier:
    L = ["| T | R | prior | KL | extra KL vs R=1 | NLL gain vs R=1 | net ELBO gain vs R=1 | predicted net (hier) | predicted extra KL | tau_l (l=0..3) | tau_2 / |z_g| (rad) | sigma2 (per-sample est) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for T in sorted({T for (T, R) in hier}):
        b = base[T]; e1 = -(b["nll_samp"] * N + b["kl"])
        for R in (2, 3, 4):
            if (T, R) in ind:
                r = ind[(T, R)]; L.append(f"| {T} | {R} | independent (v2) | {r['kl']:.0f} | {r['kl']-b['kl']:.0f} | {b['nll_samp']-r['nll_samp']:.3f} | {-(r['nll_samp']*N+r['kl'])-e1:+.0f} | | | | | {mom[(T,R)]['samp_each']['sigma2']:.4f} |" if (T, R) in mom else f"| {T} | {R} | independent (v2) | {r['kl']:.0f} | {r['kl']-b['kl']:.0f} | {b['nll_samp']-r['nll_samp']:.3f} | {-(r['nll_samp']*N+r['kl'])-e1:+.0f} | | | | | |")
            if (T, R) in hier:
                h = hier[(T, R)]; L.append(f"| {T} | {R} | hierarchical (v3) | {h['kl']:.0f} (g {h['kl_g']:.0f} + d {h['kl_d']:.0f}) | {h['kl']-b['kl']:.0f} | {b['nll_samp']-h['nll']:.3f} | {h['elbo']-e1:+.0f} | {PNET[T][R-2] if T in PNET else '-'} | {PKL[R-2]} | {', '.join(f'{t:.3f}' for t in h['tau'])} | {h['tau2_angle']:.3f} | {h['sigma2_each']:.4f} |")
    open(f"{R_}/v3_table_2p2.md", "w").write("\n".join(L) + "\n")
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for T in sorted({T for (T, R) in hier}):
        b = base[T]; e1 = -(b["nll_samp"] * N + b["kl"]); Rs = [R for R in (2, 3, 4) if (T, R) in hier]
        ax[0].plot(Rs, [hier[(T, R)]["elbo"] - e1 for R in Rs], "o-", c=col[T], label=f"T={T} hierarchical"); ax[0].plot(Rs, [-(ind[(T, R)]["nll_samp"] * N + ind[(T, R)]["kl"]) - e1 for R in Rs], "s:", c=col[T], alpha=.7)
        ax[0].plot([2, 3, 4], PNET[T], "x--", c=col[T], alpha=.6)
        ax[1].plot(Rs, [hier[(T, R)]["kl"] - b["kl"] for R in Rs], "o-", c=col[T], label=f"T={T}"); ax[1].plot(Rs, [ind[(T, R)]["kl"] - b["kl"] for R in Rs], "s:", c=col[T], alpha=.7)
        ax[2].plot(Rs, [b["nll_samp"] - hier[(T, R)]["nll"] for R in Rs], "o-", c=col[T], label=f"T={T} hier"); ax[2].plot(Rs, [b["nll_samp"] - ind[(T, R)]["nll_samp"] for R in Rs], "s:", c=col[T], alpha=.7); ax[2].plot([2, 3, 4], PNLL[T], "x--", c=col[T], alpha=.6)
    ax[1].plot([2, 3, 4], PKL, "kx--", alpha=.6)
    ax[0].set_ylabel("net ELBO gain vs R=1 (nats/config)"); ax[1].set_ylabel("extra KL vs R=1 (nats/config)"); ax[2].set_ylabel("NLL gain vs R=1 (nats/site)")
    for a in ax: a.set_xlabel("latent resolution R"); a.set_xticks([2, 3, 4])
    ax[0].legend(fontsize=6); ax[0].text(.02, .03, "solid: hierarchical, s:: independent (v2), x--: prediction", transform=ax[0].transAxes, fontsize=6)
    plt.tight_layout(); plt.savefig(f"{R_}/v3_fig_2p2.png", dpi=130); plt.close()
# ---------------- 2.3
ss = []
for f in glob.glob(f"{R_}/v3s/eval_s*.json"): ss += json.load(open(f))
if ss:
    L = ["| T | KL (subspace cov.) | KL baseline (scalar sigma) | NLL | NLL baseline | ELBO gain (nats/config) | sigma_rot | sigma_amp | sigma_bi | sigma_rot / |mu| (rad) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(ss, key=lambda r: r["T"]):
        b = base[r["T"]]; L.append(f"| {r['T']} | {r['kl']:.1f} | {b['kl']:.1f} | {r['nll']:.4f} | {b['nll_samp']:.4f} | {r['elbo']+(b['nll_samp']*N+b['kl']):+.1f} | {r['sigma_rot']:.4f} | {r['sigma_amp']:.4f} | {r['sigma_bi']:.4f} | {r['sigma_rot_angle']:.4f} |")
    open(f"{R_}/v3_table_2p3.md", "w").write("\n".join(L) + "\n")
print("ok")
