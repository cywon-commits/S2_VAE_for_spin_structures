"""E3 v3 C: regression of per-snapshot NLL gains on N*g_ring and n_def for the colour-conditioned models.  y1 = N[NLL(v1 R=1) - NLL(model)], y2 = N[NLL(v2 colour-free) - NLL(model)].
python3 cfg_regress_c.py out.json T R color"""
import sys, json, numpy as np
from cfg_regress import ols
D = "../../results/kag/"; T = float(sys.argv[2]); R = int(sys.argv[3]); color = sys.argv[4]; N = 432
z = np.load(f"{D}cfgarr_k6c_T{T:.4f}_R{R}_{color}.npz"); t = json.load(open(f"{D}cfgterms_T{T:.4f}.json")); g = N * np.array(t["g_ring"]); nd = np.array(t["n_def"], float); cl = np.repeat(np.arange(8), len(g) // 8); out = {}
for name, y in (("y1_vs_R1", z["nll_base"] - z["nll"]), ("y2_vs_colourfree", z["nll_i"] - z["nll"])):
    use = nd.std() > 0 and T > 0.0006; X = np.stack([g] + ([nd] if use else []) + [np.ones_like(g)], 1); b, s, r2 = ols(X, y, cl); keys = ["a"] + (["b"] if use else []) + ["c"]
    out[name] = dict(y_mean=float(y.mean()), r2=float(r2), **{k: float(v) for k, v in zip(keys, b)}, **{k + "_se": float(v) for k, v in zip(keys, s)}); print(T, R, color, name, out[name], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
