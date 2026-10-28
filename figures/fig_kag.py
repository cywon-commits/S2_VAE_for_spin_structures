"""Kagome figures 9-12.  Data: figures/data/kag_T*.json (compute_kag.py on data_kag/seeds, 40 seeds), results/kag/*.json (VAE evaluations)."""
import sys, glob, json, numpy as np
from style import *
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/src/kagome"); sys.path.insert(0, f"{ROOT}/src/layered"); sys.path.insert(0, f"{ROOT}/src/kag")
from matplotlib.lines import Line2D
TK = [0.0005, 0.001, 0.002, 0.004]; K = {T: json.load(open(f"{HERE}/data/kag_T{T:.4f}.json")) for T in TK}; MP = {}
TICKLAB = ["0.5", "1", "2", "4"]; TLAB = r"$T/J$ ($10^{-3}$)"
BIN_LAB = ["[0.95,1]", "[0.85,0.95)", "[0.6,0.85)", "[0,0.6)"]; BIN_KEYS = ["(0.95, 1.01)", "(0.85, 0.95)", "(0.6, 0.85)", "(0.0, 0.6)"]
A1 = np.array([1.0, 0.0]); A2 = np.array([0.5, np.sqrt(3) / 2]); SUBOFF = [np.zeros(2), A1 / 2, A2 / 2]
def site_xy(L=12):
    x, y = np.meshgrid(np.arange(L), np.arange(L), indexing="ij"); P = np.zeros((L, L, 3, 2))
    for s in range(3): P[:, :, s] = x[..., None] * A1 + y[..., None] * A2 + SUBOFF[s]
    return P
def energy_data():
    import os
    f = f"{HERE}/data/kag_energy.json"
    if os.path.exists(f): return json.load(open(f))
    from kagome_mc import energy
    out = {}
    for T in TK:
        v = []
        for fn in sorted(glob.glob(f"{ROOT}/data_kag/seeds/kag_L12_T{T:.4f}_s*.npy")): S = np.load(fn).astype(float); v.append(np.mean([energy(x) for x in S[::5]]))
        v = np.array(v); out[str(T)] = dict(E=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))), n=len(v))
    json.dump(out, open(f, "w")); return out
def psi_map(S, b=3):
    L = S.shape[0]; c = S.reshape(L // b, b, L // b, b, 3, 3).transpose(0, 2, 1, 3, 4, 5).reshape(-1, 3 * b * b, 3); Q = np.einsum('cni,cnj->cij', c, c); V = np.linalg.eigh(Q)[1]; nu = V[:, :, 0]; e1 = V[:, :, 2]; e2 = np.cross(nu, e1)
    phi = np.arctan2(np.einsum('cni,ci->cn', c, e2), np.einsum('cni,ci->cn', c, e1)); return np.abs(np.mean(np.exp(3j * phi), 1)).reshape(L // b, L // b)
def fig09():
    from ring_oracle import find_loops, build_nbrs
    from loops import global_triad
    from kagome_mc import NB
    fig = plt.figure(figsize=(W2, 2.55)); gs = fig.add_gridspec(1, 4, width_ratios=[0.78, 1.55, 0.62, 1.55], wspace=0.18)
    a = fig.add_subplot(gs[0]); E = energy_data(); y = [(E[str(T)]["E"] + 1) / T for T in TK]; e = [E[str(T)]["se"] / T for T in TK]
    for T, yy, ee in zip(TK, y, e): a.errorbar(T, yy, yerr=ee, fmt="o", color=COL_KAG[T])
    tt = np.geomspace(4e-4, 5.5e-3, 50); a.plot(tt, 0.9164 + 0.397 * np.sqrt(tt), color=OI["grey"], lw=0.8)
    a.axhline(11 / 12, color=OI["black"], ls="--", lw=0.8); a.text(0.0049, 11 / 12 - 0.004, "11/12", fontsize=6.5, ha="right", va="top")
    a.set_xscale("log"); a.set_xlim(0.0004, 0.0055); a.set_ylim(0.90, 0.96); a.set_xticks(TK); a.set_xticklabels(TICKLAB); a.minorticks_off(); a.set_xlabel(TLAB); a.set_ylabel(r"$(E+J)/T$ per site"); panel(a, "a")
    S = np.load(f"{ROOT}/data_kag/seeds/kag_L12_T0.0005_s1.npy").astype(float)[0]; X = S.reshape(-1, 3); U = global_triad(S); col = np.argmax(X @ U.T, 1).reshape(12, 12, 3); nu = np.linalg.eigh(X.T @ X)[1][:, 0]; eps = np.arcsin(np.clip(np.abs(S @ nu), 0, 1)); P = site_xy(); cc = [OI["blue"], OI["orange"], OI["green"]]
    def bonds(ax, col_="0.6", lw=0.5):
        for x in range(12):
            for yv in range(12):
                for s_ in range(3):
                    for (t, dx, dy) in NB[s_]:
                        i = (x, yv, s_); d = P[(x + dx, yv + dy, t)] - P[i] if (x + dx) < 12 and (yv + dy) < 12 and (x + dx) >= 0 and (yv + dy) >= 0 else None
                        if d is not None and np.linalg.norm(d) < 0.6: ax.plot([P[i][0], P[i][0] + d[0]], [P[i][1], P[i][1] + d[1]], color=col_, lw=lw, zorder=1)
    b = fig.add_subplot(gs[1]); bonds(b)
    for s_ in range(3): b.scatter(P[:, :, s_, 0].ravel(), P[:, :, s_, 1].ravel(), c=[cc[k] for k in col[:, :, s_].ravel()], s=2 + 55 * eps[:, :, s_].ravel() ** 2, edgecolors="k", linewidths=0.25, zorder=3)
    b.set_aspect("equal"); b.set_xticks([]); b.set_yticks([]); b.spines[:].set_visible(False); b.set_xlabel("colour: nearest triad direction\n" + r"area $\propto\varepsilon^2$", fontsize=6.3); panel(b, "b", dx=0.02, dy=0.98)
    m = fig.add_subplot(gs[2]); im = m.imshow(psi_map(S).T, origin="lower", cmap="viridis", vmin=0.5, vmax=1.0); m.set_xticks([]); m.set_yticks([]); m.set_xlabel(r"$|\psi_{cell}|$", fontsize=6.3); cb = fig.colorbar(im, ax=m, orientation="horizontal", fraction=0.08, pad=0.22, ticks=[0.5, 0.75, 1.0]); cb.ax.tick_params(labelsize=6); panel(m, "c", dx=-0.04, dy=1.02)
    c = fig.add_subplot(gs[3]); bonds(c, col_="0.75"); lp = find_loops(X, build_nbrs(12))
    for comp, ua, chi in lp:
        ln = len(comp); xs = P.reshape(-1, 2)[comp]; c.scatter(xs[:, 0], xs[:, 1], s=4, color=OI["verm"] if ln == 6 else OI["sky"], zorder=3, edgecolors="none")
    c.set_aspect("equal"); c.set_xticks([]); c.set_yticks([]); c.spines[:].set_visible(False); c.set_xlabel("two-colour loops:\nhexagons (red), longer (blue)", fontsize=6.3); panel(c, "d", dx=0.02, dy=0.98)
    savefig(fig, "fig09")
    MP["fig09"] = dict(title="Kagome overview", panels={"a": "data_kag/seeds E/site (kagome_mc.energy, every 5th snapshot, 40 seeds; s.e. over seeds) -> figures/data/kag_energy.json", "b": "data_kag/seeds/kag_L12_T0.0005_s1.npy snapshot 0 (colour via loops.global_triad; eps from the global plane normal; |psi_cell| as in local_frames.py)", "c": "same snapshot; loops: src/kag/ring_oracle.find_loops (=kagome/loops.py rule)"})
def fig10():
    fig, ax = plt.subplots(2, 2, figsize=(W2 + 1.4, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1.0])); (a, b), (c, d) = ax; Tarr = np.array(TK)
    # (a) -ln<cos3phi> and <eps^2> vs T: global frame (filled) and 3x3-cell frames without defect-touching cells (open); slopes of log-log least-squares fits in the legend
    ND = {T: json.load(open(f"{HERE}/data/kag_nodef_T{T:.4f}.json")) for T in TK}; lt = np.log(Tarr)
    sl = lambda y: np.polyfit(lt, np.log(np.array(y)), 1)[0]
    m3 = [K[T]["glob"]["minus_ln_m3"] for T in TK]; e2 = [K[T]["glob"]["eps2"] for T in TK]; m3n = [ND[T]["nodef_minus_ln_m3"] for T in TK]; e2n = [ND[T]["nodef_eps2"] for T in TK]
    a.errorbar(Tarr, [x["mean"] for x in m3], yerr=[x["se"] for x in m3], fmt="o", color=OI["blue"], label=r"$-\ln\langle\cos3\phi\rangle$, global frame (slope %.2f)" % sl([x["mean"] for x in m3]))
    a.errorbar(Tarr, [x["mean"] for x in m3n], yerr=[x["se"] for x in m3n], fmt="o", mfc="none", color=OI["blue"], label=r"$-\ln\langle\cos3\phi\rangle$, 3×3 cells, no defects (slope %.2f)" % sl([x["mean"] for x in m3n]))
    a.errorbar(Tarr, [x["mean"] for x in e2], yerr=[x["se"] for x in e2], fmt="s", color=OI["orange"], label=r"$\langle\varepsilon^2\rangle$, global frame (slope %.2f)" % sl([x["mean"] for x in e2]))
    a.errorbar(Tarr, [x["mean"] for x in e2n], yerr=[x["se"] for x in e2n], fmt="s", mfc="none", color=OI["orange"], label=r"$\langle\varepsilon^2\rangle$, 3×3 cells, no defects (slope %.2f)" % sl([x["mean"] for x in e2n]))
    t = np.array([4e-4, 5e-3]); a.plot(t, m3[0]["mean"] * (t / TK[0]) ** 1.0, color=OI["blue"], ls="--", lw=0.8, label="reference slope 1"); a.plot(t, e2[0]["mean"] * (t / TK[0]) ** 0.5, color=OI["orange"], ls=":", lw=0.9, label="reference slope 1/2")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xticks(TK); a.set_xticklabels(TICKLAB); a.minorticks_off(); a.set_xlabel(TLAB); a.set_ylabel(r"$-\ln\langle\cos3\phi\rangle$ and $\langle\varepsilon^2\rangle$"); a.legend(frameon=False, fontsize=5.2, loc="upper left"); a.set_ylim(0.04, 5.0); panel(a, "a")
    # (b) harmonic ratios by |psi_cell| bin
    x = np.arange(4)
    for T in TK:
        r6 = [K[T]["bins"][k]["r6"] for k in BIN_KEYS]; r9 = [K[T]["bins"][k]["r9"] for k in BIN_KEYS]; off = (TK.index(T) - 1.5) * 0.12
        b.errorbar(x + off, [v["mean"] for v in r6], yerr=[v["se"] for v in r6], fmt="o", color=COL_KAG[T], ms=3.5, label=f"$T={T}$"); b.errorbar(x + off, [v["mean"] for v in r9], yerr=[v["se"] for v in r9], fmt="o", mfc="none", color=COL_KAG[T], ms=3.5)
    b.axhline(4, color="k", ls=":", lw=0.8); b.axhline(9, color="k", ls=":", lw=0.8); b.text(3.45, 4.35, "4 (uniform caps)", fontsize=6, ha="right"); b.text(3.45, 8.85, "9", fontsize=6, ha="right", va="top")
    b.set_xticks(x); b.set_xticklabels(BIN_LAB, fontsize=6.5); b.set_xlabel(r"cell order $|\psi_{cell}|$ ($3\times3$ cells)"); b.set_ylabel(r"filled: $\ln m_6/\ln m_3$, open: $\ln m_9/\ln m_3$"); b.legend(frameon=False, loc="upper center", ncol=4, columnspacing=0.6, handletextpad=0.2, fontsize=5.6, bbox_to_anchor=(0.54, 1.0)); b.set_ylim(0, 12); panel(b, "b")
    # (c) <|S_tri|^2>/T vs cell eps^2
    for T in TK:
        zm = [K[T]["zeromode"][k] for k in BIN_KEYS]; c.plot([z["eps2"] for z in zm], [z["tri_over_T"] for z in zm], "o-", color=COL_KAG[T], ms=3.5, label=f"$T={T}$")
    c.axhline(11 / 4, color="k", ls="--", lw=0.8); c.text(0.335, 11 / 4 - 0.07, "11/4", fontsize=6.5, ha="right", va="top"); c.set_ylim(2.0, 3.6); c.set_xlabel(r"cell $\langle\varepsilon^2\rangle$ (bins of $|\psi_{cell}|$)"); c.set_ylabel(r"$\langle|\mathbf{S}_\triangle|^2\rangle/T$"); c.legend(frameon=False, fontsize=6, ncol=2); panel(c, "c")
    # (d) defect density per block
    w = 0.18
    for ib, (bs, ls_) in enumerate((("2", "-"), ("3", "--"))):
        for T in TK:
            dd = K[T]["defects"][bs]
            for kind, mk in (("C2", "o"), ("C3", "^"), ("Z2", "s")):
                v = dd[kind]
                if v > 0: d.plot(T, v, mk, color=COL_KAG[T], mfc=COL_KAG[T] if bs == "3" else "none", ms=4)
                else: d.plot(T, 2e-5, "v", color=COL_KAG[T], mfc="none", ms=4)
    d.set_xscale("log"); d.set_yscale("log"); d.set_ylim(1e-5, 3); d.set_xticks(TK); d.set_xticklabels(TICKLAB); d.minorticks_off(); d.set_xlabel(TLAB); d.set_ylabel("defect density per block loop")
    hh = [Line2D([], [], marker="o", color="k", ls="", mfc="none", ms=4), Line2D([], [], marker="^", color="k", ls="", mfc="none", ms=4), Line2D([], [], marker="s", color="k", ls="", mfc="none", ms=4), Line2D([], [], marker="o", color="k", ls="", ms=4), Line2D([], [], marker="v", color="k", ls="", mfc="none", ms=4)]
    d.legend(hh, [r"½-disclination of the normal (C$_2$)", r"⅓-vortex of the triad (C$_3$)", r"Z$_2$", "filled: $3\\times3$ blocks, open: $2\\times2$", "none observed (plotted at floor)"], frameon=False, fontsize=5.6, loc="upper left", bbox_to_anchor=(1.02, 1.0)); panel(d, "d")
    fig.tight_layout(); savefig(fig, "fig10")
    MP["fig10"] = dict(title="Three layers of the site distribution", panels={"a": "filled: figures/data/kag_T*.json glob (compute_kag.py: local_frames.py with b=12 frame); open: figures/data/kag_nodef_T*.json (compute_kag_nodef.py: 3x3-cell frames excluding cells that touch a defective 2x2 block loop); 40 seeds, s.e. over seeds; slopes = least-squares fits of ln y vs ln T over the four temperatures", "b": "kag_T*.json bins (defect_check.py bins, 3x3 cells)", "c": "kag_T*.json zeromode (zeromode_check.py)", "d": "kag_T*.json defects (block_defects.py rule, blocks 2x2 and 3x3; densities per block loop)"})
def lbins(length):
    edges = [(6, 7), (8, 13), (14, 25), (26, 61), (61, 1000)]; return [(lo, hi, (length >= lo) & (length < hi)) for lo, hi in edges]
def fig11():
    fig, ax = plt.subplots(2, 2, figsize=(W2, 5.0)); (a, b), (c, d) = ax; Ts = [0.0005, 0.001, 0.002, 0.004]
    for T in Ts:
        L = np.array(K[T]["loops"]["length"]); chi = np.array(K[T]["loops"]["chi"]); rms = np.array(K[T]["loops"]["rms"]); ls, ys, es, ys_s = [], [], [], []
        for lo, hi, s in lbins(L):
            if s.sum() < 20: continue
            ls.append(L[s].mean()); ys.append(np.mean(chi[s] ** 2)); es.append(np.std(chi[s] ** 2) / np.sqrt(s.sum()))
        ls = np.array(ls); ys = np.array(ys); es = np.array(es)
        a.errorbar(ls, ys, yerr=es, fmt="o", color=COL_KAG[T], ms=3.5, label=f"$T={T}$"); a.plot(ls, 23 * np.sqrt(T) / ls, color=COL_KAG[T], lw=0.8, ls="--"); b.errorbar(ls, ys * ls / np.sqrt(T), yerr=es * ls / np.sqrt(T), fmt="o", color=COL_KAG[T], ms=3.5)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(r"loop length $\ell$"); a.set_ylabel(r"$\langle\bar\chi^2\rangle$ (rad$^2$)"); a.legend(frameon=False, fontsize=6, title=r"dashed: $c\sqrt{T}/\ell$, $c=23$", title_fontsize=6); panel(a, "a")
    b.axhline(23, color="k", ls="--", lw=0.8); b.set_xscale("log"); b.set_xlabel(r"loop length $\ell$"); b.set_ylabel(r"$\langle\bar\chi^2\rangle\,\ell/\sqrt{T}$"); b.set_ylim(0, 45); panel(b, "b")
    # (c) hexagon chi distribution
    from scipy.stats import norm
    for T in Ts[:3]:
        L = np.array(K[T]["loops"]["length"]); chi = np.array(K[T]["loops"]["chi"])[L == 6]; sd = chi.std(); kurt = np.mean(chi ** 4) / np.mean(chi ** 2) ** 2
        h, e = np.histogram(chi / sd, bins=40, range=(-5, 5), density=True); c.plot(0.5 * (e[1:] + e[:-1]), h, "-", color=COL_KAG[T], lw=1.0, label=f"$T={T}$, kurt. {kurt:.1f}")
    xs = np.linspace(-5, 5, 200); c.plot(xs, norm.pdf(xs), "k--", lw=0.8, label="Gaussian (kurt. 3)"); c.set_yscale("log"); c.set_ylim(1e-4, 1); c.set_xlabel(r"hexagon $\bar\chi/\sigma$"); c.set_ylabel("density"); c.legend(frameon=False, fontsize=6); panel(c, "c")
    # (d) coherence
    for T in Ts[:3]:
        L = np.array(K[T]["loops"]["length"]); chi = np.array(K[T]["loops"]["chi"]); rms = np.array(K[T]["loops"]["rms"]); xs_, ys_ = [], []
        for lo, hi, s in lbins(L):
            if s.sum() < 20: continue
            xs_.append(np.mean(rms[s] ** 2 / L[s])); ys_.append(np.mean(chi[s] ** 2))
        d.plot(xs_, ys_, "o-", color=COL_KAG[T], ms=3.5, label=f"$T={T}$")
    lim = [1e-4, 3e-2]; d.plot(lim, lim, "k:", lw=0.8); d.set_xscale("log"); d.set_yscale("log"); d.set_xlabel(r"in-loop residual variance $/\ell$"); d.set_ylabel(r"$\langle\bar\chi^2\rangle$"); d.legend(frameon=False, fontsize=6, title="dotted: $y=x$", title_fontsize=6); panel(d, "d")
    fig.tight_layout(); savefig(fig, "fig11")
    MP["fig11"] = dict(title="Zero-mode loop rotations", panels={"a-d": "figures/data/kag_T*.json loops (compute_kag.py: loops.py rule on data_kag/seeds, every 4th snapshot, 40 seeds); c uses hexagons only (l=6)"})
def fig12():
    fig, ax = plt.subplots(1, 2, figsize=(W1 * 1.9, 2.6)); a, b = ax
    P = {T: load(f"results/kag/eval_T{T:.4f}_R1_p.json")[0]["latent_power_fraction"] for T in TK}
    for l, mk in zip(range(4), ("o", "s", "^", "D")): a.plot(TK, [P[T][l] for T in TK], mk + "-", color=[OI["grey"], OI["verm"], OI["sky"], OI["blue"]][l], ms=4, label=f"$l={l}$")
    a.axhline(0.01, color="k", ls=":", lw=0.7); a.text(0.00045, 0.0088, "1%", fontsize=6, va="top"); a.set_xscale("log"); a.set_yscale("log"); a.set_xticks(TK); a.set_xticklabels(TICKLAB); a.minorticks_off(); a.set_xlabel(TLAB); a.set_ylabel("latent power fraction ($R=1$)"); a.legend(frameon=False, fontsize=6, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handletextpad=0.3); a.set_ylim(1e-3, 1.2); panel(a, "a")
    Ls = [1, 2, 3, 4, 6, 8]
    for T in (0.0005, 0.002):
        r = load(f"results/kag/oracle_dec_T{T}.json")["nll_by_L"]; b.plot(Ls, [r[str(L)] for L in Ls], "o-", color=COL_KAG[T], ms=4, label=f"$T={T}$")
    b.set_xticks(Ls); b.set_xlabel("multipole order $L$ (refit)"); b.set_ylabel("NLL per site (decoder-frame refit)"); b.legend(frameon=False, fontsize=6.5); panel(b, "b")
    fig.tight_layout(); savefig(fig, "fig12")
    MP["fig12"] = dict(title="Octupole as the minimal sufficient representation", panels={"a": "results/kag/eval_T*_R1_p.json latent_power_fraction (REPORT_E3)", "b": "results/kag/oracle_dec_T0.0005.json, oracle_dec_T0.002.json (src/kag/oracle_dec.py; only these two temperatures were computed, 40 validation snapshots, in-sample refit)"})
def fig13():
    from fig_kag_models import load_models, decode_example, logp_from_lam
    import matplotlib.gridspec as gsp
    fig = plt.figure(figsize=(W2, 6.2)); gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
    # ---- (a) waterfall at T=0.0005 (NLL per site, lower is better)
    a = fig.add_subplot(gs[0, 0]); ev = load("results/kag/evalk6u_T0.0005_nolatent.json"); eo = load("results/kag/evalk6u_T0.0005_oracle.json")
    lg1 = open(f"{ROOT}/results/layered_gain.txt").read(); lg2 = open(f"{ROOT}/results/layered_gain2.txt").read()
    import re
    A_, B_ = [float(x) for x in re.search(r"T=0.0005: configs=\d+\s+NLL/site A=(-?[\d.]+) B=(-?[\d.]+)", lg1).groups()]; dBC = float(re.search(r"T=0.0005: loops/config [\d.]+ \(hexagons [\d.]+\)\s+B->C all loops : ([\d.]+)/site", lg2).group(1)); nloop = float(re.search(r"T=0.0005: loops/config ([\d.]+)", lg2).group(1))
    C_ = B_ - dBC; Cc = C_ + 0.5 * nloop / 432
    labels = ["colour only\n(iv, VAE)", "A global\ncaps", "B cell-frame\ncaps", "C + loop\nrotation", "VAE (ii)"]; vals = [ev["nll"], A_, B_, C_, eo["nll"]]; cols = [COL_CH["colour"], COL_CH["director"], COL_CH["local"], COL_CH["ring"], COL_CH["vae"]]
    for i, (v, c_) in enumerate(zip(vals, cols)): a.bar(i, v, 0.62, color=c_)
    a.plot([3], [Cc], "D", color="w", mec=COL_CH["ring"], ms=4.5, zorder=5, label="C, overfit-corrected"); a.plot([4], [eo["nll"] + eo["kl"] / 432], "D", color="w", mec="k", ms=4.5, zorder=5, label="VAE: NLL + KL/432"); a.legend(frameon=False, fontsize=5.4, loc="lower left")
    a.errorbar([0, 4], [ev["nll"], eo["nll"]], yerr=[ev["nll_se"], eo["nll_se"]], fmt="none", color="k", lw=0.7, capsize=2)
    for i in (1, 2, 3): a.text(i, -0.1, f"+{432 * (vals[i - 1] - vals[i]):.0f}", ha="center", va="top", fontsize=6, color="w", fontweight="bold")
    s6 = [-1.2138, -1.6949, -1.9490]                                 # same fits on seeds 1-6 (REPORT_FIGURES_v2, B-3 (a)); a.plot([1, 2, 3], s6, "_", color="k", ms=14, mew=1.6, zorder=6, label="same fits, six atypically ordered runs"); a.legend(frameon=False, fontsize=5.4, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, columnspacing=0.8)
    a.axhline(0, color="k", lw=0.6); a.set_xticks(range(5)); a.set_xticklabels(labels, fontsize=5.8); a.set_ylabel("NLL per site"); a.set_ylim(-2.2, 0.2); a.set_title("white numbers: nats/config gained vs previous bar", fontsize=6); panel(a, "a", dx=-0.12)
    # ---- (b) defect channel at T=0.002
    b = fig.add_subplot(gs[0, 1]); T = 0.002; terms = load("results/kag/cfgterms_T0.0020.json"); nd = np.array(terms["n_def"], float); g = 432 * np.array(terms["g_ring"]); rg = load("results/kag/regress_k6u_T0.0020.json")
    arr = np.load(f"{ROOT}/results/kag/cfgarr_k6u_T0.0020_R4_nolatent.npz"); arr2 = np.load(f"{ROOT}/results/kag/cfgarr_k6u_T0.0020_R4_oracle.npz"); y = arr["nll"] - arr2["nll"]; z = np.load(f"{ROOT}/results/kag/cfgarr_k6_T0.0020_R4.npz"); y0 = z["nll_base"] - z["nll"]
    rng = np.random.default_rng(0); jit = lambda v: v + 0.12 * rng.standard_normal(len(v))
    b.scatter(jit(nd), y, s=5, color=OI["blue"], alpha=0.5, lw=0, label=r"latent gain given colours, (iv)$-$(ii)"); b.scatter(jit(nd), y0, s=5, color=OI["orange"], alpha=0.5, lw=0, label=r"colour-free v2: $R{=}1-(\mathrm{i})$")
    xs = np.array([nd.min(), nd.max()])
    for k, yy, col_ in (("(iv)-(ii) NLL [main]", y, OI["blue"]), ("R=1 - (i) colour-free v2", y0, OI["orange"])):
        r = rg[k]; b.plot(xs, r["a"] * g.mean() + r["b"] * xs + r["c"], color=col_, lw=1.2, label=f"fit: {r['b']:.1f}$\\pm${r['b_se']:.1f} nats per defect")
    b.set_xlabel("3×3-block defects per snapshot ($T=0.002$)"); b.set_ylabel("per-snapshot gain (nats)"); b.set_ylim(150, 1150); b.legend(frameon=False, fontsize=5.6, loc="upper left", ncol=1); panel(b, "b")
    # ---- (c) colour conditioning
    gs_c = gs[1, 0].subgridspec(1, 2, width_ratios=[1.3, 0.95], wspace=0.15); c = fig.add_subplot(gs_c[0]); vs = [("one-hot\n(v3)", load("results/kag/evalk6c_T0.0005_R4_oracle.json")["decoded_mean_vector_len"]), ("shuffled\n$u_c$", load("results/kag/evalk6u_T0.0005_shuffled.json")["mean_vector_len"]), ("$u_c$\nonly", ev["mean_vector_len"]), ("$u_c$ +\nlatent", eo["mean_vector_len"])]
    c.bar(range(4), [v[1] for v in vs], 0.62, color=[OI["grey"], OI["grey"], OI["blue"], OI["blue"]]); c.axhline(1, color="k", lw=0.5); c.set_xticks(range(4)); c.set_xticklabels([v[0] for v in vs], fontsize=5.3); c.set_ylabel("mean-vector length of decoded\nsite distribution"); c.set_ylim(0, 1.05); panel(c, "c", dx=-0.25)
    gs_s = gs_c[1].subgridspec(2, 1, hspace=0.05); models = load_models(); lam0, lam1, Xc, Uc, frame = decode_example(models); pos = 4; usite = Uc[pos]; same = (Uc @ usite) > 0.9
    Y1 = None; vmax = None
    for k, (lam, ttl) in enumerate(((lam0, "no colour"), (lam1, "with $u_c$"))):
        axs = fig.add_subplot(gs_s[k], projection="mollweide"); im, pm = plot_sphere_density(axs, logp_from_lam(lam), frame=frame, vmax=vmax); vmax = vmax or pm
        V = Xc[same] @ frame; axs.scatter(np.arctan2(V[:, 1], V[:, 0]), np.arcsin(np.clip(V[:, 2], -1, 1)), s=5, color="w", edgecolors="k", linewidths=0.3, zorder=5); axs.set_title(ttl, fontsize=6, pad=1)
    # ---- (d) channel map
    d = fig.add_subplot(gs[1, 1]); d.axis("off"); rows = [("colouring", "$\\leq\\ln 3$ per site", "only as gauge-free\ninput $u_c$", OI["green"]), ("global plane + triad", f"{432 * (ev['nll'] - A_):.0f} nats/cfg", "yes ($R{=}1$, $l{=}3$)", OI["green"]),
        ("local frame", f"{432 * (A_ - B_):.0f} nats/cfg\n(~12 per cell)", "partly\n(not isolated)", OI["yellow"]), ("loop rotation", f"~{(432 * dBC - 0.5 * nloop) / nloop:.1f} nats/loop", "no\n($a=0.16\\pm0.04$)", OI["grey"]),
        ("topological defects", "15–23 nats/defect", "yes ($b>0$)", OI["green"]), ("gauge bits", "$G\\leq\\ln 2$", "not measured", OI["grey"])]
    d.set_xlim(0, 1); d.set_ylim(0, len(rows) + 0.7); hy = len(rows) + 0.25
    d.text(0.0, hy, "channel", fontsize=6.3, fontweight="bold"); d.text(0.34, hy, "information", fontsize=6.3, fontweight="bold"); d.text(0.66, hy, "encoded?", fontsize=6.3, fontweight="bold")
    for i, (n_, g_, v_, col_) in enumerate(rows):
        y_ = len(rows) - 1 - i; d.add_patch(plt.Rectangle((0.64, y_ + 0.06), 0.36, 0.86, color=col_, alpha=0.35, lw=0)); d.text(0.0, y_ + 0.49, n_, fontsize=5.6, va="center"); d.text(0.34, y_ + 0.49, g_, fontsize=5.6, va="center"); d.text(0.66, y_ + 0.49, v_, fontsize=5.5, va="center")
    panel(d, "d", dx=-0.02, dy=0.98)
    savefig(fig, "fig13")
    MP["fig13"] = dict(title="Kagome information-channel map", panels={"a": "waterfall: (iv),(ii): results/kag/evalk6u_T0.0005_*.json (validation seeds); A,B: results/layered_gain.txt, C: results/layered_gain2.txt (src/layered, recomputed on the repository seeds, in-sample pooled fits; overfit 0.5 nat/loop as in the chat script); chat values A=-1.214, B=-1.695, C=-1.949 differ by 0.28-0.37 nats/site (recomputed values used)", "b": "results/kag/cfgarr_k6u_T0.0020_R4_{nolatent,oracle}.npz, cfgarr_k6_T0.0020_R4.npz, cfgterms_T0.0020.json, regress_k6u_T0.0020.json", "c": "results/kag/evalk6c_T0.0005_R4_oracle.json, evalk6u_T0.0005_{shuffled,nolatent,oracle}.json; spheres: decode of kag6_T0.0005_R4.pt and kag6u_T0.0005_R4_oracle.pt (checkpoints not in git), validation snapshot 0, cell 5, site 4", "d": "schematic from REPORT_E3_v2/v4 and layered_gain*.txt"})
