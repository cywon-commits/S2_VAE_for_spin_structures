import numpy as np
def ana(path):
    d = np.load(path); lw = d["logw"]; out = []
    for j, h in enumerate(d["h"]):
        l = lw[:, j] - lw[:, j].max(); w = np.exp(l); s = w.sum(); ess = s * s / (w * w).sum()
        out.append(dict(h=float(h), m=float((w * d["mproj"][:, j]).sum() / s), m_corr=float((w * d["mproj"][:, j]).sum() / s - d["m0"].mean()), nuh2=float((w * d["nuh2"]).sum() / s), ess=float(ess)))
    return out
if __name__ == "__main__":
    import sys
    for r in ana(sys.argv[1]): print(f"h={r['h']:8.1e} m={r['m']:.5f} mc={r['m_corr']:.5f} m/(h/6)={r['m']/(r['h']/6):7.2f} nuh2={r['nuh2']:.3f} ESS={r['ess']:.1f}")
