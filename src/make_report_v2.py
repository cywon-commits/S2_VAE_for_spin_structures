"""v2 figures and tables from results/v2p (stage A) and results/v2B (stage B).  python3 make_report_v2.py"""
import json, glob, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R_ = "../results"
orc = {(r["T"], r["st"]): r for r in json.load(open(f"{R_}/oracle.json"))}; orcv = {(r["T"], r["st"]): r for r in json.load(open(f"{R_}/oracle_val.json"))}
A = [json.load(open(f)) for f in sorted(glob.glob(f"{R_}/v2p/evalA_*R1*.json"))]
Bs = [json.load(open(f)) for f in sorted(glob.glob(f"{R_}/v2B/evalA_*.json"))]
key = lambda r: (r["T"], "r" if "o_R" not in r["run"] and "r_R" in r["run"] else "o")
sig = lambda a2, a4: -(35 / 18) * a4 / a2 ** 2 if a2 > 0.3 else None
# chat predictions (REQUEST v2, sections 0 and 3)
PS2 = {0.80: [0.033, 0.024, 0.016], 0.95: [0.045, 0.033, 0.021], 1.05: [0.049, 0.028, 0.009]}
PKL = {0.80: [24, 70, 141], 0.95: [24, 69, 139], 1.05: [22, 68, 137], 1.10: [23, 70, 145]}
PNET = {0.80: [123, 269, 375], 0.95: [127, 257, 360], 1.05: [95, 235, 343], 1.10: [114, 268, 407]}
PNLL = {T: [(PNET[T][i] + PKL[T][i]) / 1728 for i in range(3)] for T in PNET}
PNLL.update({0.80: [0.085, 0.20, 0.30], 0.95: [0.087, 0.19, 0.29], 1.05: [0.068, 0.18, 0.28]})
MF = {r["T"]: r["a2_MF"] for (T_, st), r in orc.items() if st == "o"}
A1 = {r["T"]: r for r in A if "o_R1" in r["run"]}
Tl = sorted({r["T"] for r in Bs})
rows = []   # stage B table
for T in Tl:
    base = A1[T]; row = dict(T=T, R1=base)
    for R in (2, 3, 4):
        m = [r for r in Bs if r["T"] == T and r["R"] == R]
        if m: row[R] = m[0]
    rows.append(row)
# ---------------- tables
L = ["| T | stage | KL | NLL(samp) | NLL oracle(val) | a2 samp / mean-z | a2 oracle | a4 samp / mean-z | a4 oracle | mean-z vs samp a2 | odd/even | amort. gap | pass |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in sorted(A, key=lambda r: (r["T"], r["run"])):
    c = r["criteria"]; tag = "r" if "r_R1" in r["run"] else "o"
    L.append(f"| {r['T']}{tag} | A (R=1) | {r['kl']:.1f} | {r['nll_samp']:.4f} | {r['nll_oracle_val']:.4f} | {r['a2_samp']:.3f} / {r['a2_mean']:.3f} | {r['a2_oracle']:.3f} | {r['a4_samp']:+.3f} / {r['a4_mean']:+.3f} | {r['a4_oracle']:+.3f} | {100*r['a2_rel_diff_mean_vs_samp']:.1f}% | {r['odd_even']:.0e} | {r['amort_gap_mean']:.2f} | {sum(c.values())}/5 |")
open(f"{R_}/v2_table_stageA.md", "w").write("\n".join(L) + "\n")
L = ["| T | R | KL | extra KL vs R=1 | budget extra KL | NLL gain vs R=1 | predicted | a2 samp (mean-z) | MF 6S/T | sigma2 samp (mean-z) | predicted sigma2 | amort. gap |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for row in rows:
    T = row["T"]; b = row["R1"]
    L.append(f"| {T} | 1 | {b['kl']:.0f} | 0 | 0 | 0 | 0 | {b['a2_samp']:.3f} ({b['a2_mean']:.3f}) | {MF[T]:.3f} | {sig(b['a2_samp'], b['a4_samp']):.3f} ({sig(b['a2_mean'], b['a4_mean']):.3f}) | {orc[(T,'o')]['sigma2']:.3f} (oracle) | {b['amort_gap_mean']:.1f} |")
    for i, R in enumerate((2, 3, 4)):
        if R not in row: continue
        r = row[R]; ps = PS2.get(T, [None] * 3)[i]
        L.append(f"| {T} | {R} | {r['kl']:.0f} | {r['kl']-b['kl']:.0f} | {PKL[T][i]} | {b['nll_samp']-r['nll_samp']:.3f} | {PNLL[T][i]:.3f} | {r['a2_samp']:.3f} ({r['a2_mean']:.3f}) | {MF[T]:.3f} | {sig(r['a2_samp'], r['a4_samp']):.3f} ({sig(r['a2_mean'], r['a4_mean']):.3f}) | {'-' if ps is None else ps} | {r['amort_gap_mean']:.1f} |")
open(f"{R_}/v2_table_stageB.md", "w").write("\n".join(L) + "\n")
L = ["| T | dNLL L=1 (VAE frame) | oracle | L=2 | oracle | L=4 | oracle | a2 | oracle a2 | a4 | oracle a4 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in sorted(A, key=lambda r: (r["T"], r["run"])):
    fo = r["frame_oracle"]; o = orcv[(r["T"], "r" if "r_R1" in r["run"] else "o")]
    L.append(f"| {r['T']} | {fo['dNLL']['1']:.3f} | {o['dNLL']['1']:.3f} | {fo['dNLL']['2']:.4f} | {o['dNLL']['2']:.4f} | {fo['dNLL']['4']:.4f} | {o['dNLL']['4']:.4f} | {fo['a2']:.3f} | {o['a2']:.3f} | {fo['a4']:+.3f} | {o['a4']:+.3f} |")
open(f"{R_}/v2_table_f_frame_oracle.md", "w").write("\n".join(L) + "\n")
# ---------------- figures
col = {0.80: "C0", 0.95: "C1", 1.05: "C2", 1.10: "C3"}
fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
for T in Tl:
    Rs = [1] + [R for R in (2, 3, 4) if any(r["T"] == T and r["R"] == R for r in Bs)]
    get = lambda f: [f(rows[Tl.index(T)]["R1"] if R == 1 else rows[Tl.index(T)][R]) for R in Rs]
    ax[0].plot(Rs, get(lambda r: sig(r["a2_samp"], r["a4_samp"])), "o-", c=col[T], label=f"T={T} VAE (sampled z)")
    ax[0].plot(Rs, get(lambda r: sig(r["a2_mean"], r["a4_mean"])), "s:", c=col[T], alpha=.6)
    if T in PS2: ax[0].plot([1, 2, 3, 4], [orc[(T, "o")]["sigma2"]] + PS2[T], "x--", c=col[T], alpha=.6)
    b = rows[Tl.index(T)]["R1"]
    ax[1].plot(Rs[1:], [b["nll_samp"] - rows[Tl.index(T)][R]["nll_samp"] for R in Rs[1:]], "o-", c=col[T], label=f"T={T}"); ax[1].plot([2, 3, 4], PNLL[T], "x--", c=col[T], alpha=.6)
    ax[2].plot(Rs[1:], [rows[Tl.index(T)][R]["kl"] - b["kl"] for R in Rs[1:]], "o-", c=col[T], label=f"T={T}"); ax[2].plot([2, 3, 4], PKL[T], "x--", c=col[T], alpha=.6)
    ax[3].plot(Rs, get(lambda r: r["a2_samp"]), "o-", c=col[T], label=f"T={T}"); ax[3].axhline(MF[T], c=col[T], ls=":", lw=1)
ax[0].set_ylabel(r"$\sigma^2=-\frac{35}{18}a_4/a_2^2$"); ax[1].set_ylabel("NLL gain vs R=1 (nats/site)"); ax[2].set_ylabel("extra KL vs R=1 (nats/config)"); ax[3].set_ylabel(r"$a_2^{eff}$ (dotted: mean field $6S/T$)")
for a in ax: a.set_xlabel("latent resolution R"); a.set_xticks([1, 2, 3, 4])
ax[0].legend(fontsize=6); ax[1].text(.02, .95, "solid: measured, x--: chat prediction", transform=ax[1].transAxes, fontsize=6, va="top"); plt.tight_layout(); plt.savefig(f"{R_}/v2_fig_stageB.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for r in sorted(A, key=lambda r: r["T"]):
    pass
for f in sorted(glob.glob(f"{R_}/v2/v2_T*_lr3e-3.pt")) + sorted(glob.glob(f"{R_}/v2p/v2_T*_R1_b1_pruned.pt")):
    ck = torch.load(f, weights_only=False); lg = np.array(ck["log"]); nm = f.split("/")[-1][3:-3]
    if lg.size == 0: continue
    ax[0].plot(lg[:, 0], lg[:, 4], label=nm, lw=1); ax[1].plot(lg[:, 0], lg[:, 3], label=nm, lw=1)
ax[0].set_yscale("log"); ax[0].set_ylabel("val KL (nats/config)"); ax[1].set_ylabel("val NLL (nats/site)"); ax[1].set_ylim(1.4, 2.0)
ax[2].plot([r["T"] for r in sorted(A, key=lambda r: r["T"])], [r["kl"] for r in sorted(A, key=lambda r: r["T"])], "ko-"); ax[2].set_ylabel("converged KL (R=1)"); ax[2].set_xlabel("T/J")
for a in ax[:2]: a.set_xlabel("step"); a.legend(fontsize=5)
plt.tight_layout(); plt.savefig(f"{R_}/v2_fig_stageA_curves.png", dpi=130); plt.close(); print("ok")
