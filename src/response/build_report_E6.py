import json, glob, os, numpy as np
from ana_C6 import load as loadC
R = "/home/user/S2_VAE_for_spin_structures"; RS = f"{R}/results/response"
A = json.load(open(f"{RS}/E6_A.json")); SITE = {0.95: dict(B=0.129, Bs=0.048, Q=0.39, Qs=0.098), 1.05: dict(B=0.212, Bs=0.071, Q=0.90, Qs=0.134)}
ta = ["| model | T | R | χ_Q dec | χ_Q lat | χ_Q VAE | χ_Q FDT | ratio Q | χ_B dec | χ_B lat | χ_B VAE | χ_B FDT (site) | ratio B | gen-check χ_B (60 z) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in A:
    x = r["resp"]; s = SITE[r["T"]]; q = x["Q_dec"] + x["Q_lat"]; b = x["B_dec"] + x["B_lat"]
    ta.append(f"| {r['model']} ({r['run'][:3]}) | {r['T']} | {r['R']} | {x['Q_dec']:.3f} | {x['Q_lat']:.2f} | {q:.2f} | {s['Q']:.2f} | {q/s['Q']:.2f} | {x['B_dec']:.3f} | {x['B_lat']:.3f} ± {x['B_lat_se']:.3f} | {b:.3f} | {s['B']:.3f} ({s['Bs']:.3f}) | {b/s['B']:.2f} | {r['gen']['B_gen']:.3f} |")
tb = ["| model | T | R | χ_B lat, full | global only (deviations removed) | + biaxial dev. | + rotation dev. | + rotation+amplitude dev. | χ_Q lat full / global only / +rot+amp |", "|---|---|---|---|---|---|---|---|---|"]
for r in A:
    sp = r["split"]; tb.append(f"| {r['model']} | {r['T']} | {r['R']} | {r['resp']['B_lat']:.3f} | {sp['none']['B_lat']:.3f} | {sp['bi']['B_lat']:.3f} | {sp['rot']['B_lat']:.3f} | {sp['rotamp']['B_lat']:.3f} | {r['resp']['Q_lat']:.2f} / {sp['none']['Q_lat']:.2f} / {sp['rotamp']['Q_lat']:.2f} |")
tc = []
for T in (0.95, 1.05):
    C = loadC(T); tc.append(f"\n**T = {T}** (4 seeds, random start)\n\n| D | S | ⟨P₂(d·e)⟩ | ⟨B⟩ |\n|---|---|---|---|")
    for D, v in C.items(): tc.append(f"| {D:+.3f} | {v['S'][0]:.3f} ± {v['S'][1]:.3f} | {v['A'][0]:.3f} ± {v['A'][1]:.3f} | {v['B'][0]:.3f} ± {v['B'][1]:.3f} |")
sl = ["| T | easy-axis slope dS/dD (D=0.01→0.05) | easy-plane slope dS/d|D| (0.01→0.05) = χ_Q/2 | 2× easy-plane | χ_Q FDT |", "|---|---|---|---|---|"]
for T in (0.95, 1.05):
    C = loadC(T); a = (C[0.05]["S"][0] - C[0.01]["S"][0]) / 0.04; b = (C[-0.05]["S"][0] - C[-0.01]["S"][0]) / 0.04 * -1 * -1; b = (C[-0.05]["S"][0] - C[-0.01]["S"][0]) / 0.04
    sl.append(f"| {T} | {a:.2f} | {b:.2f} | {2*b:.2f} | {SITE[T]['Q']:.2f} |")
hm = ["| model | R | D | S (HMC) | ⟨P₂(d·e)⟩ | ⟨B⟩ | acc. | τ_int | chain-sd (S/A/B) | S (reweight, ESS) |", "|---|---|---|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob(f"{RS}/E6_B_*.json")):
    d = json.load(open(f)); rw = {r["D"]: r for r in d.get("reweight", [])}
    for r in sorted(d["hmc"], key=lambda r: r["D"]):
        w = rw.get(r["D"]); hm.append(f"| {d['run'][:3]} T={d['T']} | {d['R']} | {r['D']:+.3f} | {r['S']:.3f} | {r['A']:.3f} | {r['B']:.3f} | {r['acc']:.2f} | {r['tau_int']:.1f} | {r['S_chain_sd']:.3f}/{r['A_chain_sd']:.3f}/{r['B_chain_sd']:.3f} | " + (f"{w['S']:.3f} ({w['ess']:.0f})" if w else "–") + " |")
txt = open(f"{R}/src/response/report_E6_template.md").read()
for k, v in (("@@TA@@", ta), ("@@TB@@", tb), ("@@TC@@", tc), ("@@SL@@", sl), ("@@HM@@", hm)): txt = txt.replace(k, "\n".join(v))
open(f"{R}/REPORT_E6.md", "w").write(txt)
