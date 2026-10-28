import json, os, time, torch
from e6_response import *
REF = json.load(open(f"{HERE}/fdt_ll.json")); SITE = {0.95: dict(B=0.129, Bse=0.002, Bsite=0.048, Q=0.39, Qsite=0.098), 1.05: dict(B=0.212, Bse=0.006, Bsite=0.071, Q=0.90, Qsite=0.134)}
@torch.no_grad()
def gen_B(m, z, nd=5, gen=None):
    beta = 1 / m.T; mult = (12 // m.R) ** 3; p = probs(m.decode_cells(z)); acc = []; accQ = []
    for i in range(len(p)):
        for _ in range(nd):
            X = XQ[torch.multinomial(p[i].repeat_interleave(mult, 0), 1, generator=gen).squeeze(1)]; Qt = 1.5 * (X.T @ X) / N - 0.5 * torch.eye(3, dtype=torch.float64); w, V = torch.linalg.eigh(Qt); d = V[:, 2]
            e1 = torch.tensor([1.0, 0, 0], dtype=torch.float64) - d[0] * d; e1 = e1 / e1.norm(); e2 = torch.cross(d, e1, dim=0); a, b = X @ e1, X @ e2
            acc.append(torch.stack([(a ** 2 - b ** 2).sum(), (2 * a * b).sum()])); accQ.append((1.5 * (X @ d) ** 2 - 0.5).mean())
    A = torch.stack(acc); Sq = torch.stack(accQ)
    return dict(B_gen=float(0.5 * beta / N * ((A[:, 0] - A[:, 0].mean()) ** 2 + (A[:, 1] - A[:, 1].mean()) ** 2).mean()), Q_gen=float(beta * N * Sq.var()))
if __name__ == "__main__":
    runs = [(f"results/v3h/v3h_T{T:.2f}o_R{R}_b1.pt", T, R, "A") for T in (0.95, 1.05) for R in (2, 3, 4)] + [(f"results/v5a/v5a_T1.05o_R{R}_b1.pt", 1.05, R, "B") for R in (2, 3, 4)]; out = []
    for p, T, R, mod in runs:
        m = load_nem(f"{ROOT}/{p}"); g = torch.Generator().manual_seed(0); t0 = time.time(); z = torch.cat([posterior_z(m, val_spins(T), g) for _ in range(8)]); s = SITE[T]
        rec = dict(run=os.path.basename(p), model=mod, T=T, R=R, fdt=s, resp=response_ch(m, z), gen=gen_B(m, z[:60], gen=g), post_first60=response_ch(m, z[:60]))
        rec["split"] = {kp: response_ch(m, split_deviations(m, z, kp)) for kp in ("none", "bi", "rot", "rotamp")}
        zp = sample_prior(m, 1000, g); rec["prior"] = response_ch(m, zp); out.append(rec); r = rec["resp"]
        print(rec["run"], "Q", round(r["Q_dec"] + r["Q_lat"], 3), "B", round(r["B_dec"] + r["B_lat"], 3), "dec", round(r["B_dec"], 3), "gen", rec["gen"], f"{time.time()-t0:.0f}s", flush=True)
        json.dump(out, open(f"{ROOT}/results/response/E6_A.json", "w"), indent=1)
