"""Task B (nematic): reweight prior samples to the quadrupole-field ensemble -eps sum P2(n.d), random lab d per sample.  -> results/response/B_nem.json"""
import sys, json, os, time, numpy as np, torch
from nem_response import *
EPS = [0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
@torch.no_grad()
def run(path, K=4000, ch=200):
    m = load_nem(path); beta = 1 / m.T; mult = (12 // m.R) ** 3; g = torch.Generator().manual_seed(5); z = sample_prior(m, K, g)
    d = torch.randn(K, 3, generator=g, dtype=torch.float64); d /= d.norm(dim=-1, keepdim=True); Y2 = sh([2], d)                        # [K,5]
    logw = torch.zeros(K, len(EPS), dtype=torch.float64); S = torch.zeros_like(logw)
    for i in range(0, K, ch):
        lam = m.decode_cells(z[i:i + ch]).double(); k = len(lam); f = 1.5 * (d[i:i + k] @ XQ.T) ** 2 - 0.5; LZ0 = torch.logsumexp(lam @ YQ.T + LWQ, -1)
        for j, e in enumerate(EPS):
            lh = lam.clone(); lh[..., 3:8] += (beta * e * 4 * math.pi / 5) * Y2[i:i + k, None, :]; a = lh @ YQ.T + LWQ
            logw[i:i + k, j] = mult * (torch.logsumexp(a, -1) - LZ0).sum(1); S[i:i + k, j] = mult * torch.einsum("kcq,kq->k", torch.softmax(a, -1), f) / N
    out = []
    for j, e in enumerate(EPS):
        l = logw[:, j] - logw[:, j].max(); w = l.exp(); s = w.sum(); out.append(dict(eps=e, S=float((w * S[:, j]).sum() / s), ess=float(s * s / (w * w).sum())))
    return out
if __name__ == "__main__":
    res = {}
    for p in ("results/v3h/v3h_T0.95o_R2_b1.pt", "results/v3h/v3h_T1.05o_R2_b1.pt", "results/v3h/v3h_T0.95o_R4_b1.pt", "results/v3h/v3h_T1.05o_R4_b1.pt", "results/v5a/v5a_T1.05o_R2_b1.pt", "results/v5a/v5a_T1.05o_R4_b1.pt"):
        res[os.path.basename(p)[:-3]] = run(f"{ROOT}/{p}"); print(p, [(r["eps"], round(r["S"], 3), round(r["ess"])) for r in res[os.path.basename(p)[:-3]]], flush=True)
        json.dump(res, open(f"{ROOT}/results/response/B_nem.json", "w"), indent=1)
