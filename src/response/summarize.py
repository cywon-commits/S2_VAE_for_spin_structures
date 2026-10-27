import json, numpy as np, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = collections.defaultdict(list)
for d in map(json.loads, open("out/scan.jsonl")): D[(d["T"], d["h"])].append(d)
summ = []
for (T, h), L in sorted(D.items()):
    a = lambda k: np.array([x[k] for x in L])
    m = a("m"); summ.append(dict(T=T, h=h, n=len(L), m=m.mean(), m_se=m.std(ddof=1)/np.sqrt(len(m)) if len(m)>1 else 0,
        chi_zz=a("chi_zz_fluct").mean(), chi_xx=a("chi_xx_fluct").mean(), nu_h2=a("nu_h2").mean(), oop=a("oop").mean(), e=a("e").mean()))
json.dump(summ, open("out/scan_summary.json", "w"), indent=1)
print(f"{'T':>6} {'h':>5} {'n':>2} {'m':>8} {'se':>7} {'m/(h/6)':>8} {'chi_zz':>7} {'chi_xx':>7} {'nu.h^2':>7}")
for s in summ:
    print(f"{s['T']:6.3f} {s['h']:5.2f} {s['n']:2d} {s['m']:8.5f} {s['m_se']:7.5f} {s['m']/(s['h']/6):8.4f} {s['chi_zz']:7.3f} {s['chi_xx']:7.3f} {s['nu_h2']:7.3f}")
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
hh = np.linspace(0, 6.5, 200)
for T, c in ((0.001, "C0"), (0.005, "C3")):
    S = [s for s in summ if s["T"] == T]; h = np.array([s["h"] for s in S])
    ax[0].plot(h, [s["m"] for s in S], "o", ms=3.5, color=c, label=f"MC, T/J = {T}")
    ax[1].plot(h, [6*s["chi_zz"] for s in S], "o-", ms=3.5, color=c, label=f"T/J = {T}")
    ax[2].plot(h, [s["nu_h2"] for s in S], "o-", ms=3.5, color=c, label=f"T/J = {T}")
ax[0].plot(hh, np.minimum(hh/6, 1), "k--", lw=0.8, label="T = 0: m = h/6J")
ax[0].set_xlabel("h / J"); ax[0].set_ylabel("m"); ax[0].legend(frameon=False, fontsize=8, loc="lower right")
ax[1].axhline(1, color="gray", lw=0.6, ls=":"); ax[1].set_ylim(-0.05, 1.3)
ax[1].set_xlabel("h / J"); ax[1].set_ylabel(r"$\chi_{\parallel}\,/\,(1/6J)$ (fluctuation)"); ax[1].legend(frameon=False, fontsize=8, loc="lower left")
ax[2].axhline(1/3, color="gray", lw=0.6, ls=":"); ax[2].text(3.6, 0.35, "isotropic 1/3", fontsize=8, color="gray")
ax[2].set_xlabel("h / J"); ax[2].set_ylabel(r"$\langle(\hat\nu\cdot\hat h)^2\rangle$"); ax[2].legend(frameon=False, fontsize=8, loc="center right")
for a, l in zip(ax, "abc"): a.text(-0.17, 1.02, f"({l})", transform=a.transAxes, fontsize=10)
fig.tight_layout(); fig.savefig("out/kag_mh.png", dpi=150)
