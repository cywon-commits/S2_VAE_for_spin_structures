"""v5 tables.  python3 make_report_v5.py"""
import json, glob
R_ = "../results"; N = 1728; T = 1.05
base = [json.load(open(f)) for f in glob.glob(f"{R_}/v2p/evalA_*T1.05*R1*.json")][0]; e1 = -(base["nll_samp"] * N + base["kl"])
B1 = [r for r in json.load(open(f"{R_}/v4s2/eval6_allch.json")) if r["T"] == T][0]; e1b = B1["elbo"]
up = {r["R"]: r for f in glob.glob(f"{R_}/v4u/eval5_*.json") for r in json.load(open(f)) if r["T"] == T}
w = {}
for f in glob.glob(f"{R_}/v4w/eval7_*.json") + glob.glob(f"{R_}/v4w2/eval7_*.json"):
    for r in json.load(open(f)):
        if r["T"] == T and not (r["R"] == 4 and "act31" not in r["run"]): w[r["R"]] = r
ab = {r["R"]: r for f in glob.glob(f"{R_}/v5a/eval8_*.json") for r in json.load(open(f))}
sb = {r["R"]: r for f in glob.glob(f"{R_}/v5b/eval8_*.json") for r in json.load(open(f))}
PRED = {2: 42, 3: 101, 4: 140}; BUD_DA = {2: 21, 3: 62, 4: 124}
L = ["| R | model | NLL gain vs R=1 | extra KL vs R=1 | net ELBO gain vs R=1 | predicted net ELBO | extra KL / (dir+amp budget) |", "|---|---|---|---|---|---|---|"]
for R in (2, 3, 4):
    rows = [("isotropic up/down (v4)", up[R]["nll"], up[R]["kl"], up[R]["elbo"]), ("subspace covariance (v4)", w[R]["nll"], w[R]["kl"], w[R]["elbo"]),
            ("1.3 ablation: no biaxial deviation", ab[R]["nll"], ab[R]["kl"], ab[R]["elbo"]), ("1.4 separated decoder + subspace", sb[R]["nll"], sb[R]["kl"], sb[R]["elbo"])]
    for name, nll, kl, el in rows:
        L.append(f"| {R} | {name} | {base['nll_samp']-nll:.3f} | {kl-base['kl']:.0f} | {el-e1:+.0f} | {PRED[R]:+d} | {(kl-base['kl'])/BUD_DA[R]:.1f}x |")
open(f"{R_}/v5_table_13_14.md", "w").write("\n".join(L) + f"\n\n(R=1 baseline: NLL {base['nll_samp']:.4f}, KL {base['kl']:.1f}, ELBO {e1:.1f}; separated-decoder R=1 model: NLL {B1['nll']:.4f}, KL {B1['kl']:.1f}, ELBO {e1b:.1f})\n")
kd = {r["R"]: r for r in json.load(open(f"{R_}/v5/kl_decomp_T1.05.json"))}
L = ["| R | cell KL (exact) | rotation | amplitude | biaxial | other channels | rot+amp | dir+amp budget | biaxial + other | residual (decomp. vs exact) |", "|---|---|---|---|---|---|---|---|---|---|"]
for R in (2, 3, 4):
    r = kd[R]; ps = r["per_subspace"]; ra = ps["rotation"] + ps["amplitude"]
    L.append(f"| {R} | {r['kl_cell_exact']:.1f} | {ps['rotation']:.1f} | {ps['amplitude']:.1f} | {ps['biaxial']:.1f} | {r['other_channels']:.1f} | {ra:.1f} | {BUD_DA[R]} | {ps['biaxial']+r['other_channels']:.1f} | {r['residual']:.1f} |")
L.append("\nmean terms / variance terms per subspace:\n\n| R | rotation (mean / var) | amplitude (mean / var) | biaxial (mean / var) |\n|---|---|---|---|")
for R in (2, 3, 4):
    r = kd[R]; L.append(f"| {R} | " + " | ".join(f"{r['mean_terms'][n]:.1f} / {r['var_terms'][n]:.1f}" for n in ('rotation', 'amplitude', 'biaxial')) + " |")
open(f"{R_}/v5_table_11.md", "w").write("\n".join(L) + "\n")
am = {r["R"]: r for r in json.load(open(f"{R_}/v5/amort_T1.05.json"))}
L = ["| R | reconstruction (encoder -> direct) | KL (encoder -> direct) | ELBO gain (nats/config) | KL change / KL |", "|---|---|---|---|---|"]
for R in (2, 3, 4):
    r = am[R]; L.append(f"| {R} | {r['rec_encoder']:.1f} -> {r['rec_direct']:.1f} ({r['rec_change']:+.1f}) | {r['kl_encoder']:.1f} -> {r['kl_direct']:.1f} ({r['kl_change']:+.1f}) | {r['elbo_gain']:.1f} | {100*r['kl_change']/r['kl_encoder']:.0f}% |")
open(f"{R_}/v5_table_12.md", "w").write("\n".join(L) + "\n")
l3 = json.load(open(f"{R_}/v5/l3_check.json"))
L = ["| model | T | R | power fraction l=3 | NLL change when l=3 is zeroed | when all odd l zeroed |", "|---|---|---|---|---|---|"]
for r in l3: L.append(f"| {r['run'].split('_')[0]} | {r['T']} | {r['R']} | {100*r['power_fraction_l1_to_l6'][2]:.2f}% | {r['dNLL_l3']:+.1e} | {r['dNLL_odd']:+.1e} |")
open(f"{R_}/v5_table_15.md", "w").write("\n".join(L) + "\n"); print("ok")
