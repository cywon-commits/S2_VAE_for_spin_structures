import json, os, time
from qresp import *
runs = [(f"results/v5a/v5a_T1.05o_R{R}_b1.pt", 1.05, R, "B") for R in (2, 3, 4)] + [(f"results/v3h/v3h_T{T:.2f}o_R{R}_b1.pt", T, R, "A") for T in (1.05, 0.95) for R in (2, 3, 4)]
out = {}
for p, T, R, mod in runs:
    m = load_nem(f"{ROOT}/{p}"); g = torch.Generator().manual_seed(0); t0 = time.time(); z = torch.cat([posterior_z(m, val_spins(T), g) for _ in range(4)]); dec, lat = vae_chiq(m, z)
    name = os.path.basename(p)[:-3]; np.savez(f"{ROOT}/results/response/E7_q_{name}.npz", dec=dec, lat=lat, T=T, R=R)
    print(name, "dec", np.round(comb(dec[:, None, None, None], dec).T[0], 3) if False else np.round(dec, 3), "lat(q=0)", np.round(lat[:, 0, 0, 0], 3), f"{time.time()-t0:.0f}s", flush=True)
