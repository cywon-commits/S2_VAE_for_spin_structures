"""v4 tables and figures.  python3 make_report_v4.py"""
import json, glob, re, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R_ = "../results"; N = 1728
base = {r["T"]: r for r in (json.load(open(f)) for f in glob.glob(f"{R_}/v2p/evalA_*R1*.json"))}
ind = {(r["T"], r["R"]): r for r in (json.load(open(f)) for f in glob.glob(f"{R_}/v2B/evalA_*.json"))}
hier = {}
for f in glob.glob(f"{R_}/v3h/eval_h*.json"):
    for r in json.load(open(f)): hier[(r["T"], r["R"])] = r
up = {}
for f in glob.glob(f"{R_}/v4u/eval5_*.json"):
    for r in json.load(open(f)): up[(r["T"], r["R"])] = r
fs = {}
for f in glob.glob(f"{R_}/v4u/frame_sigma_*.json"):
    for r in json.load(open(f)): fs[(r["T"], r["R"])] = r
ORC = {}   # cell-frame oracle gains (this work, 100 configs, shuffled-spin overfit removed)
for T in (0.95, 1.05):
    try:
        for line in open(f"{R_}/cell_gain_T{T:.2f}.txt"):
            mm = re.search(r"R=(\d) .* net ([0-9.]+)", line)
            if mm: ORC[(T, int(mm.group(1)))] = float(mm.group(2))
    except FileNotFoundError: pass
CHAT_ORC = {0.95: {2: .038, 3: .086, 4: .142}}
PKL = {2: 19, 3: 53, 4: 102}; PNET = {0.95: {2: 56, 3: 110, 4: 148}, 1.05: {2: 42, 3: 101, 4: 140}, 0.80: {2: 55, 3: 117, 4: 155}, 1.10: {2: 50, 3: 116, 4: 169}}
PS2 = {0.80: [0.046, 0.040, 0.031, 0.024], 0.95: [0.062, 0.052, 0.040, 0.028], 1.05: [0.068, 0.055, 0.035, 0.015]}
Ts = sorted({T for (T, R) in up})
# ---- task A table
L = ["| T | R | NLL gain | oracle gain (cell frames) | gain / oracle | extra KL | budget | extra KL / budget | net ELBO gain | predicted | tau_2 (rad = tau_2/\\|z_g\\|) | dev angle \\|mu_c-mu_g\\|/\\|z_g\\| | v2 independent: gain / extra KL / net | v3 hierarchical: gain / extra KL / net |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for T in Ts:
    b = base[T]; e1 = -(b["nll_samp"] * N + b["kl"])
    for R in (2, 3, 4):
        if (T, R) not in up: continue
        u = up[(T, R)]; g = b["nll_samp"] - u["nll"]; o = ORC.get((T, R)); i = ind[(T, R)]; h = hier.get((T, R)); ostr = "-" if o is None else f"{o:.3f}"; rstr = "-" if o is None else f"{100*g/o:.0f}%"
        hs = f"{b['nll_samp']-h['nll']:.3f} / {h['kl']-b['kl']:.0f} / {h['elbo']-e1:+.0f}" if h else "-"
        L.append(f"| {T} | {R} | {g:.3f} | {ostr} | {rstr} | {u['kl']-b['kl']:.0f} | {PKL[R]} | {(u['kl']-b['kl'])/PKL[R]:.1f}x | {u['elbo']-e1:+.0f} | {PNET[T][R]:+d} | {u['tau2_angle']:.3f} | {u['dev_angle']:.3f} | {b['nll_samp']-i['nll_samp']:.3f} / {i['kl']-b['kl']:.0f} / {-(i['nll_samp']*N+i['kl'])-e1:+.0f} | {hs} |")
open(f"{R_}/v4_table_A.md", "w").write("\n".join(L) + "\n")
# ---- task C table (decoder-frame oracle)
L = ["| T | R | sigma2 (cell-axes frame oracle) | predicted sigma2 | a_loc | decoder NLL (mean-z) | frame-oracle NLL | sigma2 (decoder intrinsic, per-sample est.) |", "|---|---|---|---|---|---|---|---|"]
for T in sorted({T for (T, R) in fs}):
    for R in (1, 2, 3, 4):
        if (T, R) not in fs: continue
        r = fs[(T, R)]; ps = PS2.get(T, [None] * 4)[R - 1]; dv = up.get((T, R)); dsv = f"{dv['sigma2_each']:.4f}" if dv else "-"
        L.append(f"| {T} | {R} | {r['sigma2']:.4f} | {'-' if ps is None else ps} | {r['a_loc']:.3f} | {r['nll_decoder_meanz']:.4f} | {r['nll_frame_oracle']:.4f} | {dsv} |")
open(f"{R_}/v4_table_C.md", "w").write("\n".join(L) + "\n")
# ---- figures
col = {0.80: "C0", 0.95: "C1", 1.05: "C2", 1.10: "C3"}
fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
for T in Ts:
    b = base[T]; e1 = -(b["nll_samp"] * N + b["kl"]); Rs = [R for R in (2, 3, 4) if (T, R) in up]
    ax[0].plot(Rs, [b["nll_samp"] - up[(T, R)]["nll"] for R in Rs], "o-", c=col[T], label=f"T={T} v4 (up/down)"); ax[0].plot(Rs, [b["nll_samp"] - ind[(T, R)]["nll_samp"] for R in Rs], "s:", c=col[T], alpha=.6)
    ax[0].plot([R for R in Rs if (T, R) in hier], [b["nll_samp"] - hier[(T, R)]["nll"] for R in Rs if (T, R) in hier], "^--", c=col[T], alpha=.5); ax[0].plot([R for R in Rs if (T, R) in ORC], [ORC[(T, R)] for R in Rs if (T, R) in ORC], "k+", ms=9)
    ax[1].plot(Rs, [up[(T, R)]["kl"] - b["kl"] for R in Rs], "o-", c=col[T], label=f"T={T}"); ax[1].plot(Rs, [ind[(T, R)]["kl"] - b["kl"] for R in Rs], "s:", c=col[T], alpha=.6)
    ax[1].plot([R for R in Rs if (T, R) in hier], [hier[(T, R)]["kl"] - b["kl"] for R in Rs if (T, R) in hier], "^--", c=col[T], alpha=.5)
    ax[2].plot(Rs, [up[(T, R)]["elbo"] - e1 for R in Rs], "o-", c=col[T], label=f"T={T}"); ax[2].plot(Rs, [-(ind[(T, R)]["nll_samp"] * N + ind[(T, R)]["kl"]) - e1 for R in Rs], "s:", c=col[T], alpha=.6)
    ax[2].plot([R for R in Rs if (T, R) in hier], [hier[(T, R)]["elbo"] - e1 for R in Rs if (T, R) in hier], "^--", c=col[T], alpha=.5); ax[2].plot(Rs, [PNET[T][R] for R in Rs], "x--", c=col[T], alpha=.5)
    ax[3].plot([R for R in [1] + Rs if (T, R) in fs], [fs[(T, R)]["sigma2"] for R in [1] + Rs if (T, R) in fs], "o-", c=col[T], label=f"T={T}")
    if T in PS2: ax[3].plot([1, 2, 3, 4], PS2[T], "x--", c=col[T], alpha=.5)
ax[1].plot([2, 3, 4], [PKL[R] for R in (2, 3, 4)], "kx--", alpha=.6)
ax[0].set_ylabel("NLL gain vs R=1 (nats/site)"); ax[1].set_ylabel("extra KL vs R=1 (nats/config)"); ax[2].set_ylabel("net ELBO gain vs R=1 (nats/config)"); ax[3].set_ylabel(r"$\sigma^2$ (cell-axes frame oracle)")
for a in ax: a.set_xlabel("latent resolution R"); a.set_xticks([1, 2, 3, 4] if a is ax[3] else [2, 3, 4])
ax[0].legend(fontsize=6); ax[0].text(.02, .03, "o-: v4 up/down; s:: v2 indep.; ^--: v3 hier.; +: oracle; x--: prediction", transform=ax[0].transAxes, fontsize=5.5)
plt.tight_layout(); plt.savefig(f"{R_}/v4_fig_A.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for f in sorted(glob.glob(f"{R_}/v4u/v4u_T0.95o_R*_b1.pt")):
    ck = torch.load(f, weights_only=False); lg = np.array(ck["log"]); nm = f.split("/")[-1].split("_")[2]
    if lg.size: ax[0].plot(lg[:, 0], lg[:, 4], label=nm, lw=1); ax[1].plot(lg[:, 0], lg[:, 3], label=nm, lw=1)
ax[0].set_ylabel("val KL (nats/config), T=0.95"); ax[1].set_ylabel("val NLL (nats/site), T=0.95"); ax[0].set_yscale("log")
for a in ax: a.set_xlabel("step"); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{R_}/v4_fig_curves.png", dpi=130); plt.close(); 
# ---- task B table
ctrl = {r["T"]: r for r in json.load(open(f"{R_}/v4c/eval_c.json"))}
allch = {r["T"]: r for r in json.load(open(f"{R_}/v4s2/eval6_allch.json"))}
L = ["| T | decoder | KL | mean term (1/2 sum mu^2) | variance term | NLL | ELBO | active l=2 channels: |mu| | sigma |", "|---|---|---|---|---|---|---|---|---|"]
for T in (0.95, 1.05):
    b = base[T]
    L.append(f"| {T} | v2 pruned (baseline, trained 40k+10k) | {b['kl']:.1f} | - | - | {b['nll_samp']:.4f} | {-(b['nll_samp']*N+b['kl']):.1f} | - | - |")
    c = ctrl[T]; L.append(f"| {T} | plain decoder re-initialised (control, 25k) | {c['kl']:.1f} | {c['kl_mean_term']:.1f} | {c['kl_var_term']:.1f} | {c['nll']:.4f} | {c['elbo']:.1f} | {[round(x,2) for x in c['mu_norm']]} | {[round(x,3) for x in c['sigma']]} |")
    a = allch[T]; L.append(f"| {T} | direction/magnitude separated, all active l=2 channels (25k) | {a['kl']:.1f} | {a['kl_mean_term']:.1f} | {a['kl_var_term']:.1f} | {a['nll']:.4f} | {a['elbo']:.1f} | {[round(x,2) for x in a['mu_norm']]} | {[round(x,3) for x in a['sigma']]} |")
open(f"{R_}/v4_table_B.md", "w").write("\n".join(L) + "\n")
# ---- subspace-covariance table
w = {}
for f in glob.glob(f"{R_}/v4w/eval7_*.json") + glob.glob(f"{R_}/v4w2/eval7_*.json"):
    for r in json.load(open(f)): w[(r["T"], r["R"])] = r
L = ["| T | R | isotropic (v4 up/down): extra KL / net ELBO | subspace covariance: extra KL / net ELBO | predicted net ELBO | tau rot / amp / bi | tau_rot as angle (rad) |", "|---|---|---|---|---|---|---|"]
for (T, R), r in sorted(w.items()):
    b = base[T]; e1 = -(b["nll_samp"] * N + b["kl"]); u = up[(T, R)]
    L.append(f"| {T} | {R} | {u['kl']-b['kl']:.0f} / {u['elbo']-e1:+.0f} | {r['kl']-b['kl']:.0f} / {r['elbo']-e1:+.0f} | {PNET[T][R]:+d} | {', '.join(f'{x:.2f}' for x in r['tau_rot_amp_bi'])} | {r['tau_rot_angle']:.3f} |")
open(f"{R_}/v4_table_W.md", "w").write("\n".join(L) + "\n")
print("ok")
