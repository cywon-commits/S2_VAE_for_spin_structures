"""E3 v2 evaluation of site-resolved models (kag6_*) against the R=1 v1 baseline, paired by validation seed; saves per-config arrays for the regression (C).
python3 eval_k6.py out.json T R [tag]"""
import sys, json, numpy as np, torch
from kmodel import kag_features
from kmodel6 import KVAE6, site_Y
from eval_k import decoder_moments, harmonic_ratios
from eval_ku import per_config as per_config_v1
from kdata import load_T, VAL_SEEDS
torch.set_num_threads(1)
@torch.no_grad()
def per_config6(path, nsamp=2, CH=20):
    ck = torch.load(path, weights_only=False); a = ck["args"]; R = a["R"]; m = KVAE6(R=R, order=131); m.load_state_dict({k: v for k, v in ck["state"].items() if not k.startswith("logz.")}, strict=False); m.eval()
    S = torch.tensor(load_T(a["T"], VAL_SEEDS)); F = kag_features(S, R); mu_g, lv_g = m.encode(F.mean(1, keepdim=True)); mu_c, lv_c = m.encode(F); kl_g, kl_c = m.kl_terms(mu_g, lv_g, mu_c, lv_c)
    torch.manual_seed(0); nll = torch.zeros(len(S)); mom = []
    for _ in range(nsamp):
        for i in range(0, len(S), CH):
            sl = slice(i, i + CH); lam = m.decode(m.sample(mu_c[sl], lv_c[sl])); nll[sl] += m.nll_cfg(lam, site_Y(S[sl], R)) / nsamp; mom.append(decoder_moments(lam))      # moments pooled over the cell's sites
    cat = {k: torch.cat([d[k].reshape(-1) for d in mom]) for k in mom[0]}
    return dict(nll_cfg=nll, kl_g=kl_g, kl_c=kl_c, harm=harmonic_ratios(cat), tau=m.log_tau.exp().tolist(), args=a)
def run(T, R, tag=""):
    base = per_config_v1(f"../../results/kag/kag_T{T:.4f}_R1_p.pt"); r = per_config6(f"../../results/kag/kag6_T{T:.4f}_R{R}{tag}.pt")
    nseed = len(VAL_SEEDS); n = len(base["nll_cfg"]); per = n // nseed; seedm = lambda v: v.reshape(nseed, per).mean(1); se = lambda v: v.std().item() / np.sqrt(nseed)
    e_b = -(base["nll_cfg"] + base["kl_g"] + base["kl_c"]); e_r = -(r["nll_cfg"] + r["kl_g"] + r["kl_c"]); gain = seedm(e_r - e_b); nllg = seedm(base["nll_cfg"] - r["nll_cfg"]) / 432; klx = seedm(r["kl_g"] + r["kl_c"] - base["kl_g"] - base["kl_c"])
    np.savez(f"../../results/kag/cfgarr_k6_T{T:.4f}_R{R}{tag}.npz", nll=r["nll_cfg"].numpy(), kl=(r["kl_g"] + r["kl_c"]).numpy(), nll_base=base["nll_cfg"].numpy(), kl_base=(base["kl_g"] + base["kl_c"]).numpy())
    return dict(T=T, R=R, net_elbo_gain=gain.mean().item(), net_elbo_gain_se=se(gain), nll_gain=nllg.mean().item(), nll_gain_se=se(nllg), extra_kl=klx.mean().item(), extra_kl_se=se(klx), kl=(r["kl_g"] + r["kl_c"]).mean().item(),
                kl_g=r["kl_g"].mean().item(), kl_c=r["kl_c"].mean().item(), nll=(r["nll_cfg"] / 432).mean().item(), harmonic=r["harm"], tau=r["tau"], baseline=dict(nll=(base["nll_cfg"] / 432).mean().item(), kl=base["kl_g"].mean().item()))
if __name__ == "__main__":
    T = float(sys.argv[2]); R = int(sys.argv[3]); tag = sys.argv[4] if len(sys.argv) > 4 else ""; r = run(T, R, tag); json.dump(r, open(sys.argv[1], "w"), indent=1); h = r["harmonic"]
    print(f"T={T} R={R}{tag}: net ELBO gain {r['net_elbo_gain']:.1f}+-{r['net_elbo_gain_se']:.1f} | NLL {r['nll']:.3f} gain {r['nll_gain']:.4f}+-{r['nll_gain_se']:.4f} | KL {r['kl']:.1f} extra {r['extra_kl']:.1f}+-{r['extra_kl_se']:.1f} | -ln m3 {h['minus_ln_m3']:.3f} ln m6/ln m3 {h['ln_m6_over_ln_m3']:.2f} ln m9/ln m3 {h['ln_m9_over_ln_m3']:.2f} eps2 {h['eps2']:.3f}", flush=True)
