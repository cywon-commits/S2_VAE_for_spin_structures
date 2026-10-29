import json, glob, os, numpy as np
from ana_C6 import load as loadC
from ana_E7q import *
R_ = ROOT; RS = f"{ROOT}/results/response"; d = json.load(open(f"{RS}/E7_1ab.json")); SITE = {0.95: dict(B=0.129, Q=0.39), 1.05: dict(B=0.212, Q=0.90)}
t1 = ["| model | T | R | E‖z‖² ch.1 / ch.2 (N(0,I): 5) | ‖z‖ main channel (mean ± sd) | z_g comp. variances (10 active; N(0,I): 1) | cell noise ε variances (active; N(0,I): 1) |", "|---|---|---|---|---|---|---|"]
t2 = ["| model | T | R | S data | S post. | (i) z_g post, cells prior | (ii) z_g prior, cells post | old prior | refit prior (GMM k) | var ratio refit/data | var ratio (i)/data |", "|---|---|---|---|---|---|---|---|---|---|---|"]
t3 = ["| model | T | R | χ_Q (agg. post.) | χ_Q old prior | χ_Q refit prior | χ_B (agg. post.) | χ_B old prior | χ_B refit prior | MC χ_Q / χ_B |", "|---|---|---|---|---|---|---|---|---|---|"]
f2 = lambda l: "/".join(f"{x:.2f}" for x in l)
for n, r in d.items():
    t1.append(f"| {r['model']} | {r['T']} | {r['R']} | {r['ch21']['E_n2']:.2f} / {r['ch31']['E_n2']:.2f} | {r['ch31']['norm_mean']:.2f} ± {r['ch31']['norm_std']:.2f} | {f2(r['zg_var'])} | {f2(r['eps_var'])} |")
    S = r["S"]; sd = S["data"]["std"] ** 2; t2.append(f"| {r['model']} | {r['T']} | {r['R']} | {S['data']['mean']:.3f}±{S['data']['std']:.3f} | {S['post']['mean']:.3f}±{S['post']['std']:.3f} | {S['i_zg_post_cells_prior']['mean']:.3f}±{S['i_zg_post_cells_prior']['std']:.3f} | {S['ii_zg_prior_cells_post']['mean']:.3f}±{S['ii_zg_prior_cells_post']['std']:.3f} | {S['old_prior']['mean']:.3f}±{S['old_prior']['std']:.3f} | {S['new_prior']['mean']:.3f}±{S['new_prior']['std']:.3f} (k={r['gmm']['k']}) | {S['new_prior']['std']**2/sd:.1f} | {S['i_zg_post_cells_prior']['std']**2/sd:.1f} |")
    q = lambda k, c: r[k][c + "_dec"] + r[k][c + "_lat"]; s = SITE[r["T"]]
    t3.append(f"| {r['model']} | {r['T']} | {r['R']} | {q('resp_post','Q'):.2f} | {q('resp_old','Q'):.1f} | {q('resp_new','Q'):.2f} | {q('resp_post','B'):.3f} | {q('resp_old','B'):.2f} | {q('resp_new','B'):.3f} | {s['Q']:.2f} / {s['B']:.3f} |")
# 1-c comparison
C = loadC(1.05); Ds = sorted(C); t4 = []
for f in sorted(glob.glob(f"{RS}/E7_C_*.json")):
    e = json.load(open(f)); name = e["run"]; tag = " [GMM k=200]" if "k200" in f else ""; old = json.load(open(f"{RS}/E6_B_{name}.json")) if os.path.exists(f"{RS}/E6_B_{name}.json") else (json.load(open(f"{RS}/E6_B_{name}_part.json")) if os.path.exists(f"{RS}/E6_B_{name}_part.json") else None)
    oh = {r["D"]: r for r in old["hmc"]} if old else {}; nh = {r["D"]: r for r in e["hmc"]}
    t4.append(f"\n**{name}{tag}** (T={e['T']}, R={e['R']})\n\n| D | S: MC / old / refit | ⟨P₂(d·e)⟩: MC / old / refit | ⟨B⟩: MC / old / refit | acc. | τ_int | chain-sd S/A/B (refit) |\n|---|---|---|---|---|---|---|")
    for D in Ds:
        if D not in nh: continue
        o = oh.get(D); n_ = nh[D]; fo = lambda k: f"{o[k]:.3f}" if o else "–"
        t4.append(f"| {D:+.3f} | {C[D]['S'][0]:.3f} / {fo('S')} / {n_['S']:.3f} | {C[D]['A'][0]:.3f} / {fo('A')} / {n_['A']:.3f} | {C[D]['B'][0]:.3f} / {fo('B')} / {n_['B']:.3f} | {n_['acc']:.2f} | {n_['tau_int']:.1f} | {n_['S_chain_sd']:.3f}/{n_['A_chain_sd']:.3f}/{n_['B_chain_sd']:.3f} |")
# task 2 table
r2 = load_all(); t5 = ["| model | T | R | channel | χ_dec (MC site) | q=0: VAE / MC | excess ratio (VAE−site)/(pred−site), 0<k≤R/2 | same, R/2<k<R | VAE/site for k>R+1 |", "|---|---|---|---|---|---|---|---|---|"]
for n, v in sorted(r2.items(), key=lambda kv: (kv[1]["model"], -kv[1]["T"], kv[1]["R"])):
    k = np.sqrt(v["k2"]); Rr = v["R"]
    for c, ch in enumerate(["amplitude m=0", "director m=±1", "biaxial m=±2"]):
        low = (k > 0) & (k <= Rr / 2 + 1e-9); mid = (k > Rr / 2) & (k < Rr - 1e-9); hi = k > Rr + 1; ex = lambda m_: ((v["vae"][c][m_] - v["site"][c]) / (v["pred"][c][m_] - v["site"][c])).mean() if m_.any() else float("nan")
        q0 = f"{v['vae'][c][0]:.3f} / {v['mc'][c][0]:.3f}" if ch != "director m=±1" else "–"
        t5.append(f"| {v['model']} | {v['T']} | {Rr} | {ch} | {v['dec'][c]:.3f} ({v['site'][c]:.3f}) | {q0} | {ex(low):.2f} | {ex(mid):.2f} | {(v['vae'][c][hi] / v['site'][c]).mean():.2f} |")
txt = open(f"{ROOT}/src/response/report_E7_template.md").read()
for kk, vv in (("@@T1@@", t1), ("@@T2@@", t2), ("@@T3@@", t3), ("@@T4@@", t4), ("@@T5@@", t5)): txt = txt.replace(kk, "\n".join(vv))
open(f"{ROOT}/REPORT_E7.md", "w").write(txt)
