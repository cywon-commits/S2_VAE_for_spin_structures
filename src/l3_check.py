"""v5 1.5: odd-l (l=3) content of the cell decoders under posterior-sample decoding, and its contribution to the NLL.  python3 l3_check.py out.json run.pt ...
(works for MVAE5 'v4u' and MVAE7 'v4w' checkpoints)"""
import sys, json, numpy as np, torch
from mvae2 import compute_stats, cell_features
from mvae5 import MVAE5
from mvae7 import MVAE7
torch.set_num_threads(1)
SL = [(sum(2 * k + 1 for k in range(1, l)), sum(2 * k + 1 for k in range(1, l + 1))) for l in range(1, 7)]
@torch.no_grad()
def check(path, nsamp=8):
    ck = torch.load(path, weights_only=False); a = ck["args"]; R = a["R"]; up3 = a.get("up3", False)
    m = (MVAE7 if up3 else MVAE5)(L=12, R=R); m.load_state_dict(ck["state"], strict=False); m.eval(); nc = (12 // R) ** 3
    S = torch.tensor(np.load(f"{a['data']}/ll_L12_T{a['T']:.3f}_{a['st']}.npy")); nv = max(len(S) // 10, 10); Sv = S[-nv:]; F = cell_features(Sv, R); St = compute_stats(Sv, R)
    if up3: mu_c, lv_c, la_c = m.encode3(F)
    else: mu_c, lv_c = m.encode_c(F)
    torch.manual_seed(0); pw = np.zeros(6); n_full = n_even = n_l3off = 0.
    for _ in range(nsamp):
        z = m.sample3(mu_c, lv_c, la_c) if up3 else m.sample(mu_c, lv_c); lam = m.decode_cells(z)
        pw += np.array([(lam[..., s:e] ** 2).sum(-1).mean().item() for s, e in SL]) / nsamp
        nll = lambda l: ((-(l * St).sum(-1) + nc * m.logz(l)).sum(-1).mean().item()) / 1728 / nsamp
        n_full += nll(lam)
        le = lam.clone()
        for i in (0, 2, 4): le[..., SL[i][0]:SL[i][1]] = 0            # zero all odd l (1,3,5)
        n_even += nll(le)
        l3 = lam.clone(); l3[..., SL[2][0]:SL[2][1]] = 0                # zero l=3 only
        n_l3off += nll(l3)
    fr = pw / pw.sum()
    return dict(run=path.split("/")[-1][:-3], T=a["T"], R=R, power_fraction_l1_to_l6=fr.tolist(), nll_full=n_full, nll_odd_zeroed=n_even, nll_l3_zeroed=n_l3off, dNLL_odd=n_even - n_full, dNLL_l3=n_l3off - n_full)
if __name__ == "__main__":
    out = [check(p) for p in sys.argv[2:]]
    for r in out: print(r["run"], "power fractions l=1..6 %s | NLL %.4f, odd-l zeroed %+.4f, l=3 zeroed %+.4f" % ([round(x, 4) for x in r["power_fraction_l1_to_l6"]], r["nll_full"], r["dNLL_odd"], r["dNLL_l3"]), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
