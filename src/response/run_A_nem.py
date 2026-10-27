"""E5 task A (nematic).  python3 run_A_nem.py [K] -> results/response/A_nem.json"""
import sys, json, os, time, torch
from nem_response import *
K = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
REF = json.load(open(f"{HERE}/fdt_ll.json")); out = []
runs = [(f"results/v3h/v3h_T{T:.2f}o_R{R}_b1.pt", T, R, "v3h") for T in (0.95, 1.05) for R in (2, 3, 4)] + [(f"results/v5a/v5a_T1.05o_R{R}_b1.pt", 1.05, R, "v5a") for R in (2, 3, 4)]
for p, T, R, tag in runs:
    if not os.path.exists(f"{ROOT}/{p}"): continue
    m = load_nem(f"{ROOT}/{p}"); ref = REF[f"{T:g}"] if f"{T:g}" in REF else REF[str(T)]; t0 = time.time(); g = torch.Generator().manual_seed(0)
    rec = dict(run=os.path.basename(p), tag=tag, T=T, R=R, S_data=ref["S"], curie_par=ref["zeeman_curie_par"], curie_perp=ref["zeeman_curie_perp"], fdt_par=ref["zeeman_fdt_par"], fdt_perp=ref["zeeman_fdt_perp"],
               quad_fdt=ref["quad_fdt"], quad_fdt_se=ref["quad_fdt_se"], quad_weiss=ref["quad_weiss"])
    rec["prior"] = response(m, sample_prior(m, K, g)); Sv = val_spins(T); rec["posterior"] = response(m, torch.cat([posterior_z(m, Sv, g) for _ in range(8)])); out.append(rec)
    print(rec["run"], {k: (round(v["quad_dec"], 3), round(v["quad_lat"], 3), round(v["zee_dec_par"], 3)) for k, v in rec.items() if k in ("prior", "posterior")}, f"{time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(f"{ROOT}/results/response/A_nem.json", "w"), indent=1)
