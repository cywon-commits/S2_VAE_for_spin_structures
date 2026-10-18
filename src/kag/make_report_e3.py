"""Builds the markdown tables for REPORT_E3.md from results/kag/*.json."""
import json, glob, os
D = "../../results/kag/"; Ts = [0.0005, 0.001, 0.002, 0.004]
print("## A: R=1\n| T/J | NLL/site ± se(seed) | KL | active | power l0,l1,l2,l3 | -ln m3 | ln m6/ln m3 | ln m9/ln m3 | eps2 |\n|---|---|---|---|---|---|---|---|---|")
for T in Ts:
    d = json.load(open(f"{D}eval_T{T:.4f}_R1_p.json"))[0]; h = d["decoder_moments"]; p = d["latent_power_fraction"]
    print(f"| {T} | {d['nll']:.3f} ± {d['nll_se_seeds']:.3f} | {d['kl']:.1f} ± {d['kl_se_seeds']:.1f} | {d['active_components']} | {', '.join(f'{x:.3f}' for x in p)} | {h['minus_ln_m3']:.3f} | {h['ln_m6_over_ln_m3']:.2f} | {h['ln_m9_over_ln_m3']:.2f} | {h['eps2']:.3f} |")
print("\n## B: net ELBO gain vs R=1 (nats/config ± se over 8 val seeds)\n| T/J | R | net ELBO gain | NLL gain/site | extra KL | -ln m3 | ln m6/ln m3 | ln m9/ln m3 | eps2 |\n|---|---|---|---|---|---|---|---|---|")
for T in Ts:
    for R in (3, 4, 6, 12):
        f = f"{D}evalu_T{T:.4f}_R{R}.json"
        if not os.path.exists(f): continue
        d = json.load(open(f)); h = d["harmonic"]
        print(f"| {T} | {R} | {d['net_elbo_gain']:.1f} ± {d['net_elbo_gain_se']:.1f} | {d['nll_gain']:.3f} ± {d['nll_gain_se']:.3f} | {d['extra_kl']:.0f} ± {d['extra_kl_se']:.0f} | {h['minus_ln_m3']:.3f} | {h['ln_m6_over_ln_m3']:.2f} | {h['ln_m9_over_ln_m3']:.2f} | {h['eps2']:.3f} |")
print("\n## C\n| T | R | ELBO amortized | direct | gap | gap/site | KL l0 (glob/cell) | l1 | l2 | l3 |\n|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(D + "diagc_*.json")):
    d = json.load(open(f)); k = d["kl_per_l"]
    print(f"| {d['T']} | {d['R']} | {d['elbo_amort']:.1f} | {d['elbo_direct']:.1f} | {d['gap']:.1f} ± {d['gap_se']:.1f} | {d['gap_per_site']:.3f} | " + " | ".join(f"{k[f'l{l}']['glob']:.1f}/{k[f'l{l}']['cell']:.1f}" for l in range(4)) + " |")
print("\n## ring oracle\n| T | coverage | hex c | NLL none | ring | shuffled | gain ring | gain shuffled |\n|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(D + "ring_T*.json")):
    d = json.load(open(f)); print(f"| {d['T']} | {d['site_coverage']:.2f} | {d['hex_chi2_over_sqrtT_ell']:.1f} | {d['none']:.3f} | {d['ring']:.3f} | {d['shuffled']:.3f} | {d['gain_ring']:.3f} | {d['gain_shuffled']:.3f} |")
