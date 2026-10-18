"""E3 C: KL decomposition (per order l, global vs cell) and amortization gap (decoder fixed, per-config posterior optimised directly).
python3 diag_c.py out.json T R [nconf] [steps]   (R=1: results/kag/kag_T{T}_R1_p.pt ; R>1: kagu_T{T}_R{R}.pt)"""
import sys, json, numpy as np, torch
from eval_ku import load
from kmodel import kag_features, kag_stats
from kdata import load_T, VAL_SEEDS
torch.set_num_threads(2)
T = float(sys.argv[2]); R = int(sys.argv[3]); NC = int(sys.argv[4]) if len(sys.argv) > 4 else 8; STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 300
path = f"../../results/kag/kag_T{T:.4f}_R1_p.pt" if R == 1 else f"../../results/kag/kagu_T{T:.4f}_R{R}.pt"
m, a = load(path); nsub = (12 // R) ** 2; up = bool(a.get("up"))
for p in m.parameters(): p.requires_grad_(False)
S = torch.tensor(load_T(T, VAL_SEEDS)); idx = torch.arange(0, len(S), len(S) // NC)[:NC]; S = S[idx]
F = kag_features(S, R); St = kag_stats(S, R)
def elbo(mu_g, lv_g, mu_c, lv_c, ns=4):
    nll = 0
    for _ in range(ns): nll = nll + m.nll_cfg(m.decode(m.sample(mu_c, lv_c)), St, nsub) / ns
    if up: kg, kc = m.kl_terms(mu_g, lv_g, mu_c, lv_c)
    else: kg, kc = m.kl(mu_c, lv_c), torch.zeros(len(S))
    return -(nll + kg + kc), nll, kg, kc
with torch.no_grad():
    if up: mu_g, lv_g = m.encode(F.mean(1, keepdim=True))
    else: mu_g, lv_g = None, None
    mu_c, lv_c = m.encode(F); torch.manual_seed(0); e0, nll0, kg0, kc0 = elbo(mu_g, lv_g, mu_c, lv_c, 8)
    # KL decomposition per order
    lvx = m.expand_logvar(lv_c); lidx = torch.tensor(sum([[l] * (4 * (2 * l + 1)) for l in range(4)], []))
    if up:
        lvg = m.expand_logvar(lv_g); tau = m.tau_comp()
        klc = ((torch.log(tau) - 0.5 * lvx + (lvx.exp() + (mu_c - mu_g) ** 2 + lvg.exp()) / (2 * tau ** 2) - 0.5) * m.zmask)
        klg = 0.5 * (mu_g ** 2 + lvg.exp() - 1 - lvg) * m.zmask
    else:
        klc = 0.5 * (mu_c ** 2 + lvx.exp() - 1 - lvx) * m.zmask; klg = torch.zeros(1, 1, 1)
    per_l = {f"l{l}": dict(cell=klc[..., lidx == l].sum((-1, -2)).mean().item(), glob=(klg[..., lidx == l].sum((-1, -2)).mean().item() if up else 0.0)) for l in range(4)}
# amortization gap: free per-config posterior (init from encoder), Adam
P = [x.clone().requires_grad_(True) for x in ([mu_g, lv_g, mu_c, lv_c] if up else [None, None, mu_c, lv_c]) if x is not None]
opt = torch.optim.Adam(P, lr=0.02); torch.manual_seed(1)
for s in range(STEPS):
    args = (P[0], P[1], P[2], P[3]) if up else (None, None, P[0], P[1])
    e, *_ = elbo(*args, 2); loss = -e.sum() / len(S); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    args = (P[0], P[1], P[2], P[3]) if up else (None, None, P[0], P[1]); torch.manual_seed(0); e1, nll1, kg1, kc1 = elbo(*args, 8)
gap = (e1 - e0); out = dict(T=T, R=R, nconf=len(S), elbo_amort=e0.mean().item(), elbo_direct=e1.mean().item(), gap=gap.mean().item(), gap_se=gap.std().item() / np.sqrt(len(S)),
    gap_per_site=gap.mean().item() / 432, nll_a=nll0.mean().item(), nll_d=nll1.mean().item(), kl_a=(kg0 + kc0).mean().item(), kl_d=(kg1 + kc1).mean().item(), kl_per_l=per_l)
json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps(out), flush=True)
