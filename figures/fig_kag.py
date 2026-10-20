"""Kagome figures 9-12.  Data: figures/data/kag_T*.json (compute_kag.py on data_kag/seeds, 40 seeds), results/kag/*.json (VAE evaluations)."""
import sys, glob, json, numpy as np
from style import *
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/src/kagome"); sys.path.insert(0, f"{ROOT}/src/layered"); sys.path.insert(0, f"{ROOT}/src/kag")
from matplotlib.lines import Line2D
TK = [0.0005, 0.001, 0.002, 0.004]; K = {T: json.load(open(f"{HERE}/data/kag_T{T:.4f}.json")) for T in TK}; MP = {}
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
    fig = plt.figure(figsize=(W2, 3.4)); gs = fig.add_gridspec(1, 4, width_ratios=[0.8, 1.7, 0.7, 1.7], wspace=0.12)
    a = fig.add_subplot(gs[0]); E = energy_data(); y = [(E[str(T)]["E"] + 1) / T for T in TK]; e = [E[str(T)]["se"] / T for T in TK]
    for T, yy, ee in zip(TK, y, e): a.errorbar(T, yy, yerr=ee, fmt="o", color=COL_KAG[T])
    a.axhline(11 / 12, color=OI["black"], ls="--", lw=0.8); a.axhline(1.0, color=OI["grey"], ls=":", lw=0.8); a.text(0.00042, 11 / 12 - 0.012, "11/12", fontsize=6.5); a.text(0.00042, 1.004, "1", fontsize=6.5)
    a.set_xscale("log"); a.set_xlim(0.0004, 0.0055); a.set_ylim(0.85, 1.03); a.set_xticks(TK); a.set_xticklabels(["5e-4", "1e-3", "2e-3", "4e-3"], rotation=45); a.minorticks_off(); a.set_xlabel(r"$T/J$"); a.set_ylabel(r"$(E+J)/T$ per site"); panel(a, "a")
    S = np.load(f"{ROOT}/data_kag/seeds/kag_L12_T0.0005_s1.npy").astype(float)[0]; X = S.reshape(-1, 3); U = global_triad(S); col = np.argmax(X @ U.T, 1).reshape(12, 12, 3); nu = np.linalg.eigh(X.T @ X)[1][:, 0]; eps = np.arcsin(np.clip(np.abs(S @ nu), 0, 1)); P = site_xy(); cc = [OI["blue"], OI["orange"], OI["green"]]
    def bonds(ax, col_="0.6", lw=0.5):
        for x in range(12):
            for yv in range(12):
                for s_ in range(3):
                    for (t, dx, dy) in NB[s_]:
                        i = (x, yv, s_); j = ((x + dx) % 12, (yv + dy) % 12, t); d = P[(x + dx, yv + dy, t)] - P[i] if (x + dx) < 12 and (yv + dy) < 12 and (x + dx) >= 0 and (yv + dy) >= 0 else None
                        if d is not None and np.linalg.norm(d) < 0.6: ax.plot([P[i][0], P[i][0] + d[0]], [P[i][1], P[i][1] + d[1]], color=col_, lw=lw, zorder=1)
    b = fig.add_subplot(gs[1]); bonds(b)
    for s_ in range(3): b.scatter(P[:, :, s_, 0].ravel(), P[:, :, s_, 1].ravel(), c=[cc[k] for k in col[:, :, s_].ravel()], s=2 + 55 * eps[:, :, s_].ravel() ** 2, edgecolors="k", linewidths=0.25, zorder=3)
    b.set_aspect("equal"); b.set_xticks([]); b.set_yticks([]); b.spines[:].set_visible(False); b.set_title(r"colour: nearest triad direction; size $\propto\varepsilon^2$", fontsize=6.5); panel(b, "b", dx=0.0, dy=1.1)
    m = fig.add_subplot(gs[2]); im = m.imshow(psi_map(S).T, origin="lower", cmap="viridis", vmin=0.5, vmax=1.0); m.set_xticks([]); m.set_yticks([]); m.set_title(r"$|\psi_{cell}|$ ($3\times3$)", fontsize=6.5); cb = fig.colorbar(im, ax=m, fraction=0.06, pad=0.03, orientation="horizontal"); cb.ax.tick_params(labelsize=6)
    c = fig.add_subplot(gs[3]); bonds(c, col_="0.75"); lp = find_loops(X, build_nbrs(12))
    for comp, ua, chi in lp:
        ln = len(comp); xs = P.reshape(-1, 2)[comp]; c.scatter(xs[:, 0], xs[:, 1], s=4, color=OI["verm"] if ln == 6 else OI["sky"], zorder=3, edgecolors="none")
    c.set_aspect("equal"); c.set_xticks([]); c.set_yticks([]); c.spines[:].set_visible(False); c.set_title("two-colour loops: hexagons (red), longer (blue)", fontsize=6.5); panel(c, "c", dx=0.0, dy=1.1)
    savefig(fig, "fig09")
    MP["fig09"] = dict(title="Kagome overview", panels={"a": "data_kag/seeds E/site (kagome_mc.energy, every 5th snapshot, 40 seeds; s.e. over seeds) -> figures/data/kag_energy.json", "b": "data_kag/seeds/kag_L12_T0.0005_s1.npy snapshot 0 (colour via loops.global_triad; eps from the global plane normal; |psi_cell| as in local_frames.py)", "c": "same snapshot; loops: src/kag/ring_oracle.find_loops (=kagome/loops.py rule)"})
