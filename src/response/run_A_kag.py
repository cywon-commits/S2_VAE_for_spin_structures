"""E5 task A (kagome): linear susceptibility of the VAE decoders.  python3 run_A_kag.py [K]   -> results/response/A_kag.json"""
import sys, json, os, time, torch
from vae_response import *
K = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
REF = json.load(open(f"{HERE}/fdt_kag.json")); out = json.load(open(f"{ROOT}/results/response/A_kag.json")) if os.path.exists(f"{ROOT}/results/response/A_kag.json") else []; os.makedirs(f"{ROOT}/results/response", exist_ok=True)
def runs(T):
    t = f"{T:.4f}"; l = [f"kag_T{t}_R1.pt", f"kag6_T{t}_R1.pt"] + [f"kagu_T{t}_R{R}.pt" for R in (3, 4, 6, 12)] + [f"kag6_T{t}_R4.pt", f"kag6u_T{t}_R4_nolatent.pt", f"kag6u_T{t}_R4_oracle.pt"]
    return [r for r in l if os.path.exists(f"{ROOT}/results/kag/{r}")]
for T in (0.0005, 0.002):
    ref = REF[str(T)]
    for run in runs(T):
        if any(r['run'] == run for r in out): continue
        m = load_kag(f"{ROOT}/results/kag/{run}"); mult = mult_of(m); beta = 1 / m.T; rec = dict(run=run, T=T, R=m.R, kind=m.kind, fdt_in=ref["fdt_inplane"], fdt_nu=ref["fdt_normal"], weiss_in=ref["weiss_inplane"], weiss_nu=ref["weiss_normal"]); t0 = time.time()
        gen = torch.Generator().manual_seed(0)
        if m.kind != "6u" or "nolatent" in run:                                                  # ancestral prior (nolatent: u from random validation snapshots, latent unused)
            z = sample_prior(m, K, gen); u = prior_u(m, K, gen) if m.kind == "6u" else None; lam = decode(m, z, u); rec["prior"] = response_terms(lam, mult, beta)
            kc = min(60, K); rec["prior_check"] = check_generation(m, z[:kc], lam[:kc], gen=gen); rec["prior_analytic_first"] = response_terms(lam[:kc], mult, beta); del lam
        z, u = sample_posterior(m, 4, gen); lam = decode(m, z, u); rec["posterior"] = response_terms(lam, mult, beta); kc = 60; rec["posterior_check"] = check_generation(m, z[:kc], lam[:kc], gen=gen); rec["posterior_analytic_first"] = response_terms(lam[:kc], mult, beta); del lam
        out.append(rec); print(run, {k: (round(v["dec_in"] + v["lat_in"], 2), round(v["dec_nu"] + v["lat_nu"], 2)) for k, v in rec.items() if k in ("prior", "posterior")}, f"{time.time() - t0:.0f}s", flush=True)
        json.dump(out, open(f"{ROOT}/results/response/A_kag.json", "w"), indent=1)
