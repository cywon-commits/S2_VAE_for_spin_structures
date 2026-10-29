import json, os, glob, numpy as np
from qresp import *
RS = f"{ROOT}/results/response"
def load_all():
    mc = {T: np.load(f"{RS}/E7_mc_chiq_T{T}.npz") for T in (0.95, 1.05)}; res = {}
    for f in sorted(glob.glob(f"{RS}/E7_q_*.npz")):
        d = np.load(f); name = os.path.basename(f)[5:-4]; T = float(d["T"]); R = int(d["R"]); b = L // R; mcd = mc[T]
        chiMC = comb(mcd["chi"]); site = comb(mcd["chi"], mcd["site"]); dec = comb(d["dec"][:, None, None, None].repeat(L, 1).repeat(L, 2).repeat(L, 3))[:, 0, 0, 0]; lat = comb(d["lat"])
        vae = dec[:, None, None, None] + lat; F2 = form_factor2(b); pred = site[:, None, None, None] + (chiMC - site[:, None, None, None]) * F2[None]
        u, vs = shells(vae); _, ps = shells(pred); _, ms = shells(chiMC)
        res[name] = dict(T=T, R=R, model="B" if name.startswith("v5a") else "A", k2=u, vae=vs, pred=ps, mc=ms, site=site, dec=dec, lat=shells(lat)[1])
    return res
if __name__ == "__main__":
    r = load_all()
    for n, v in r.items():
        print(n, "site(MC) Q/D/B", np.round(v["site"], 3), "dec", np.round(v["dec"], 3))
        for c, ch in enumerate("QDB"):
            sel = [i for i, k2 in enumerate(v["k2"]) if k2 in (0, 1, 2, 3, 4, 6, 9, 12, 20, 36, 75)]
            print("  ", ch, " ".join(f"k2={v['k2'][i]}:{v['vae'][c][i]:.3f}/{v['pred'][c][i]:.3f}(MC {v['mc'][c][i]:.3f})" for i in sel))
