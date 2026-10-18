"""E3 B evaluation: per-config ELBO of R=1 (pruned) and R>1 up/down models on the validation seeds, paired by seed against the R=1 baseline; decoder harmonic ratios.
python3 eval_ku.py out.json T R   (R1 baseline = results/kag/kag_T{T}_R1_p.pt ; run = results/kag/kagu_T{T}_R{R}.pt)"""
import sys, json, numpy as np, torch
from kmodel import KVAE, kag_features, kag_stats
from kmodel5 import KVAE5
from eval_k import decoder_moments, harmonic_ratios
from kdata import load_T, VAL_SEEDS
torch.set_num_threads(1)
def load(path):
    ck = torch.load(path, weights_only=False); a = ck["args"]; cls = KVAE5 if a.get("up") else KVAE
    m = cls(R=a["R"], order=131); m.load_state_dict({k: v for k, v in ck["state"].items() if not k.startswith("logz.")}, strict=False); m.eval(); return m, a
@torch.no_grad()
def per_config(path, nsamp=2):
    m, a = load(path); R = a["R"]; nsub = (12 // R) ** 2; N = 432
    S = torch.tensor(load_T(a["T"], VAL_SEEDS)); F = kag_features(S, R); St = kag_stats(S, R)
    if a.get("up"):
        mu_g, lv_g = m.encode(F.mean(1, keepdim=True)); mu_c, lv_c = m.encode(F); kl_g, kl_c = m.kl_terms(mu_g, lv_g, mu_c, lv_c)
    else:
        mu_c, lv_c = m.encode(F); kl_g = m.kl(mu_c, lv_c); kl_c = torch.zeros_like(kl_g)
    torch.manual_seed(0); nll = torch.zeros(len(S)); mom = []; CH = 20
    for _ in range(nsamp):
        for i in range(0, len(S), CH):
            sl = slice(i, i + CH); lam = m.decode(m.sample(mu_c[sl], lv_c[sl])); nll[sl] += m.nll_cfg(lam, St[sl], nsub) / nsamp; mom.append(decoder_moments(lam))
    cat = {k: torch.cat([d[k].reshape(-1) for d in mom]) for k in mom[0]}
    return dict(nll_cfg=nll, kl_g=kl_g, kl_c=kl_c, harm=harmonic_ratios(cat), tau=(m.log_tau.exp().tolist() if a.get("up") else None), args=a)
def run(T, R, tag_base="_p", tag_run=""):
    base = per_config(f"../../results/kag/kag_T{T:.4f}_R1{tag_base}.pt"); r = per_config(f"../../results/kag/kagu_T{T:.4f}_R{R}{tag_run}.pt")
    nseed = len(VAL_SEEDS); n = len(base["nll_cfg"]); per = n // nseed
    e_b = -(base["nll_cfg"] + base["kl_g"] + base["kl_c"]); e_r = -(r["nll_cfg"] + r["kl_g"] + r["kl_c"]); seedm = lambda v: v.reshape(nseed, per).mean(1)
    gain = seedm(e_r - e_b); nllg = seedm(base["nll_cfg"] - r["nll_cfg"]) / 432; klx = seedm(r["kl_g"] + r["kl_c"] - base["kl_g"] - base["kl_c"])
    se = lambda v: v.std().item() / np.sqrt(nseed)
    return dict(T=T, R=R, net_elbo_gain=gain.mean().item(), net_elbo_gain_se=se(gain), nll_gain=nllg.mean().item(), nll_gain_se=se(nllg), extra_kl=klx.mean().item(), extra_kl_se=se(klx),
                kl=(r["kl_g"] + r["kl_c"]).mean().item(), kl_g=r["kl_g"].mean().item(), kl_c=r["kl_c"].mean().item(), nll=(r["nll_cfg"] / 432).mean().item(), harmonic=r["harm"], tau=r["tau"],
                baseline=dict(nll=(base["nll_cfg"] / 432).mean().item(), kl=(base["kl_g"]).mean().item(), harmonic=base["harm"]))
if __name__ == "__main__":
    T = float(sys.argv[2]); R = int(sys.argv[3]); r = run(T, R); json.dump(r, open(sys.argv[1], "w"), indent=1)
    h = r["harmonic"]; print(f"T={T} R={R}: net ELBO gain {r['net_elbo_gain']:.1f}+-{r['net_elbo_gain_se']:.1f} | NLL gain {r['nll_gain']:.4f}+-{r['nll_gain_se']:.4f} | extra KL {r['extra_kl']:.1f}+-{r['extra_kl_se']:.1f} (g {r['kl_g']:.1f}, c {r['kl_c']:.1f}) | -ln m3 {h['minus_ln_m3']:.3f} ln m6/ln m3 {h['ln_m6_over_ln_m3']:.2f} ln m9/ln m3 {h['ln_m9_over_ln_m3']:.2f} eps2 {h['eps2']:.3f} |e^i| {h['c1']:.3f} | tau {[round(t,3) for t in r['tau']]}", flush=True)
