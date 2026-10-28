import json, glob, os, numpy as np, subprocess
from ana_B import ana
R = "/home/user/S2_VAE_for_spin_structures"; RS = f"{R}/results/response"
tab = subprocess.run(["python3", f"{R}/src/response/tables_E5.py"], capture_output=True, text=True).stdout
# reweighting table
rw = ["| model | ESS≥100 window (max h) | χ_eff = m/h in window | ⟨(ν·ĥ)²⟩ at window edge | h at which m_VAE ≥ 0.5 (ESS=1, dominated by 1 sample) | MC m at that h |", "|---|---|---|---|---|---|"]
for f in sorted(glob.glob(f"{RS}/B_kag*_T0.00*_*.npz")):
    r = ana(f); ok = [x for x in r if x["ess"] >= 100]; w = ok[-1] if ok else r[0]; h5 = next((x["h"] for x in r if x["m"] >= 0.5), float("nan"))
    rw.append(f"| {os.path.basename(f)[2:-4]} | {w['h']:.1e} (ESS {w['ess']:.0f}) | {w['m_plot']/w['h']:.0f} | {w['nuh2']:.3f} | {h5:.0e} | {min(h5/6,1):.1e} |")
mc = ["| model | h | m (HMC) | ⟨(ν·ĥ)²⟩ | acceptance | τ_int(m) | chain-to-chain sd of m | #chains×obs |", "|---|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob(f"{RS}/Bmc_kag*_T0.0005_*.json")):
    for x in json.load(open(f)): mc.append(f"| {x['run'][:-3]} | {x['h']:g} | {x['m']:.4f} | {x['nuh2']:.3f} | {x['acc']:.2f} | {x['tau_int']:.1f} | {x['m_chain_sd']:.4f} | {len(x['m_chains'])}×{x['n_obs']} |")
cq = json.load(open(f"{RS}/C_quad.json")); cq.update(json.load(open(f"{RS}/C_quad_fine.json"))); ref = json.load(open(f"{R}/src/response/fdt_ll.json"))
ct = ["| T | S(ε) = ⟨P₂(n·d)⟩ at ε = 0, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2 | slope (ε=0.02–0.1) | slope (0.05–0.2) | χ_FDT (±se) | χ_W |", "|---|---|---|---|---|---|"]
for T in (0.95, 1.05, 1.1):
    rows = sorted((v["eps"], v["S"]) for v in cq.values() if v["T"] == T); e = np.array([a for a, _ in rows]); S = np.array([b for _, b in rows])
    sl = lambda lo, hi: np.polyfit(e[(e >= lo) & (e <= hi)], S[(e >= lo) & (e <= hi)], 1)[0]; rf = ref[str(T) if str(T) in ref else f"{T:g}"]
    ct.append(f"| {T} | {' / '.join(f'{b:.3f}' for _, b in rows)} | {sl(0.02,0.1):.2f} | {sl(0.05,0.2):.2f} | {rf['quad_fdt']:.2f} ± {rf['quad_fdt_se']:.2f} | {rf['quad_weiss']:.3f} |")
txt = open(f"{R}/src/response/report_template.md").read()
txt = txt.replace("@@TABLES@@", tab).replace("@@RW@@", "\n".join(rw)).replace("@@MC@@", "\n".join(mc)).replace("@@CT@@", "\n".join(ct))
open(f"{R}/REPORT_E5.md", "w").write(txt)
