"""Odd/even multipole power ratio of the decoded lambda (mean-z decode, 20 validation snapshots) for the nematic v2 models, recomputed directly with eval_A.power_ratio.
Reason: results/v2/evalA_*_R1_*.json store odd_even = 1.7e6 (inverted; the other evalA files store the correct 6e-7).  Output: figures/data/converged_oddeven.json"""
import sys, glob, json, numpy as np, torch
ROOT = "/home/user/S2_VAE_for_spin_structures"; sys.path.insert(0, f"{ROOT}/src")
import eval_A
from mvae2 import cell_features
torch.set_num_threads(2); out = []
for path in sorted(glob.glob(f"{ROOT}/results/v2/v2_T*_R1_b1_lr1e-3.pt") + glob.glob(f"{ROOT}/results/v2B/v2_T*_R[234]_b1.pt")):
    m, a = eval_A.load(path); S = torch.tensor(np.load(f"{ROOT}/data/ll_L12_T{a['T']:.3f}_{a['st']}.npy"))[-20:]
    with torch.no_grad(): mu, lv = m.encode(feat=cell_features(S, a["R"])); lam = m.decode_cells(mu); ev, od = eval_A.power_ratio(lam)
    r = dict(run=path.split("/")[-1], T=a["T"], R=a["R"], odd_even=float(od.mean() / ev.mean()), even_power=float(ev.mean()), kl=float(m.kl(mu, lv).mean())); out.append(r); print(r, flush=True)
json.dump(out, open(f"{ROOT}/figures/data/converged_oddeven.json", "w"), indent=1)
