import json
RS = "/home/user/S2_VAE_for_spin_structures/results/response"
A = json.load(open(f"{RS}/A_kag.json")); N = json.load(open(f"{RS}/A_nem.json"))
f = lambda x: f"{x:.3g}" if abs(x) < 1000 else f"{x:.0f}"
print("### kagome (per site, J=1; in-plane / normal)\n")
print("| model | prior χ_dec (in/ν) | prior χ_lat (in/ν) | post χ_dec (in/ν) | post χ_lat (in/ν) | χ_VAE post total (in/ν) | χ_FDT | χ_W | χ_VAE/χ_FDT (post, in) | gen-check post (in/ν) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in A:
    p = r.get("prior"); q = r["posterior"]
    pr = f"{f(p['dec_in'])} / {f(p['dec_nu'])}" if p else "–"; pl = f"{f(p['lat_in'])} / {f(p['lat_nu'])}" if p else "–"
    c = r["posterior_check"]
    print(f"| {r['run'][:-3]} | {pr} | {pl} | {f(q['dec_in'])} / {f(q['dec_nu'])} | {f(q['lat_in'])} / {f(q['lat_nu'])} | {f(q['dec_in']+q['lat_in'])} / {f(q['dec_nu']+q['lat_nu'])} | {r['fdt_in']:.3f} / {r['fdt_nu']:.3f} | {r['weiss_in']:.3f} / {r['weiss_nu']:.3f} | {(q['dec_in']+q['lat_in'])/r['fdt_in']:.0f} | {f(c['gen_in'])} / {f(c['gen_nu'])} |")
print("\n### nematic (per site)\n")
print("| model | latents | S_model | Zeeman χ_∥ / χ_⊥ (dec) | Curie ∥/⊥ | quad χ_dec | quad χ_lat | quad χ_VAE | χ_FDT (±se) | χ_W | χ_VAE/χ_FDT |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in N:
    for k in ("posterior", "prior"):
        p = r[k]; print(f"| {r['run'][:-3]} | {k} | {p['S_mean']:.3f} (data {r['S_data']:.3f}) | {p['zee_dec_par']:.3f} / {p['zee_dec_perp']:.3f} | {r['curie_par']:.3f} / {r['curie_perp']:.3f} | {p['quad_dec']:.3f} | {p['quad_lat']:.3g} | {p['quad_dec']+p['quad_lat']:.3g} | {r['quad_fdt']:.3f} ± {r['quad_fdt_se']:.3f} | {r['quad_weiss']:.3f} | {(p['quad_dec']+p['quad_lat'])/r['quad_fdt']:.2f} |")
