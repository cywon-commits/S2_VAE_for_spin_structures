"""Check 1 (nematic): generate spin configs from the decoder (posterior latents and prior latents), measure beta*N*Var(S_p) with S_p = mean P2(n.d), d = director of the realised Q tensor (same estimator as the MC fdt)."""
import json, os, torch
from nem_response import *
@torch.no_grad()
def gen_check(m, z, nd=2, gen=None):
    beta = 1 / m.T; mult = (12 // m.R) ** 3; lam = m.decode_cells(z); p = probs(lam); k, C, Q = p.shape; Sp = []
    for i in range(k):
        for _ in range(nd):
            idx = torch.multinomial(p[i].repeat_interleave(mult, 0), 1, generator=gen).squeeze(1); X = XQ[idx]; Qt = 1.5 * (X.T @ X) / N - 0.5 * torch.eye(3, dtype=torch.float64)
            w, V = torch.linalg.eigh(Qt); d = V[:, 2]; Sp.append((1.5 * (X @ d) ** 2 - 0.5).mean())
    Sp = torch.stack(Sp); return dict(S=float(Sp.mean()), chi_gen=float(beta * N * Sp.var()), chi_gen_se=float(beta * N * Sp.var() * math.sqrt(2 / (len(Sp) - 1))))
if __name__ == "__main__":
    out = []
    for p in ("results/v3h/v3h_T0.95o_R2_b1.pt", "results/v3h/v3h_T1.05o_R2_b1.pt", "results/v5a/v5a_T1.05o_R2_b1.pt", "results/v5a/v5a_T1.05o_R4_b1.pt"):
        m = load_nem(f"{ROOT}/{p}"); g = torch.Generator().manual_seed(1); Sv = val_spins(m.T)
        zp = torch.cat([posterior_z(m, Sv, g) for _ in range(2)])[:1000]; zq = sample_prior(m, 1000, g)
        r = dict(run=os.path.basename(p), post=gen_check(m, zp, gen=g), prior=gen_check(m, zq, gen=g), post_an=response(m, zp), prior_an=response(m, zq))
        out.append(r); print(r["run"], r["post"], round(r["post_an"]["quad_dec"] + r["post_an"]["quad_lat"], 3), r["prior"], round(r["prior_an"]["quad_dec"] + r["prior_an"]["quad_lat"], 3), flush=True)
        json.dump(out, open(f"{ROOT}/results/response/A_nem_check.json", "w"), indent=1)
