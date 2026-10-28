import sys, json, time, numpy as np
from ll_quad_mc import run
out = {}; t0 = time.time()
for T in (0.95, 1.05, 1.10):
    for eps in (0.003, 0.005, 0.0075, 0.015):
        ss = [run(12, T, eps, 400, 2000, 7000 + s + int(1000 * T)).mean() for s in range(4)]
        out[f"{T}_{eps}"] = dict(T=T, eps=eps, S=float(np.mean(ss)), se=float(np.std(ss, ddof=1) / 2), per_seed=[float(x) for x in ss]); print(T, eps, out[f"{T}_{eps}"]["S"], f"{time.time()-t0:.0f}s", flush=True)
        json.dump(out, open("/home/user/S2_VAE_for_spin_structures/results/response/C_quad_fine.json", "w"), indent=1)
