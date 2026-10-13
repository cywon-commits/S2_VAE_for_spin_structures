"""Evaluate trained runs: criteria (a)-(f) quantities.  python3 evaluate.py ../results/runs ../results/eval.json"""
import sys, glob, json, os, numpy as np, torch
from e3nn import o3
from mvae import MVAE, LAM_IRREPS, LAT_IRREPS, sh
torch.set_num_threads(1); torch.manual_seed(1234)

def directors(S):                      # S [C,L,L,L,3] -> (S_order[C], director[C,3])
    S = S.double(); Q = torch.einsum("cxyzi,cxyzj->cij", S, S) / S[0, ..., 0].numel() - torch.eye(3, dtype=S.dtype) / 3
    w, V = torch.linalg.eigh(Q); return 1.5 * w[:, -1], V[:, :, -1]

def load(path):
    ck = torch.load(path, weights_only=False); a = ck["args"]
    m = MVAE(L=12, R=a["R"]); m.load_state_dict(ck["state"]); m.eval(); return m, a

def val_data(a):
    S = torch.tensor(np.load(f"{a['data']}/ll_L12_T{a['T']:.3f}_{a['st']}.npy")); nv = max(len(S) // 10, 10)
    return S[-nv:]

@torch.no_grad()
def evaluate(path, Lcuts=(6,), nsamp=4, sample=True):
    m, a = load(path); S = val_data(a); Sord, d = directors(S)
    sl = [(sum(2 * k + 1 for k in range(1, l)), sum(2 * k + 1 for k in range(1, l + 1))) for l in range(1, 7)]
    Yd = {l: o3.spherical_harmonics(l, d.float(), normalize=True, normalization="integral") for l in range(1, 7)}
    acc = dict(a=np.zeros(7), pw=np.zeros(7), n=0, nll={L: 0. for L in Lcuts}, kl=0., klch=np.zeros(4), mupw=np.zeros(4), tr=np.zeros(4), mu_norm=np.zeros(4))
    for i in range(0, len(S), 10):
        x = S[i:i + 10]; B = len(x)
        mu, lv = m.encode(x)
        for s_ in range(nsamp if sample else 1):
            lam = m.decode(m.sample(mu, lv) if sample else mu)           # lam [B,1,48] for R=1 or [B,1728,48]
            for l in range(1, 7):
                lm = lam[..., sl[l - 1][0]:sl[l - 1][1]]
                acc["a"][l] += (lm * Yd[l][i:i + B, None, :]).sum(-1).mean(1).sum().item() / (nsamp if sample else 1)
                acc["pw"][l] += (lm ** 2).sum(-1).mean(1).sum().item() / (nsamp if sample else 1)
            for L in Lcuts:
                nflat = x.reshape(B, -1, 3); acc["nll"][L] += m.nll_sites(lam, nflat, L).mean(1).sum().item() / (nsamp if sample else 1)
        acc["kl"] += m.kl(mu, lv).sum().item()
        # per-l latent power and KL; l-block boundaries inside LAT_IRREPS (4 channels each)
        lvx = m.expand_logvar(lv); klc = 0.5 * (mu ** 2 + lvx.exp() - 1 - lvx).sum(-2)   # [B,64]
        off = 0
        for li in range(4):
            w = 4 * (2 * li + 1)
            acc["klch"][li] += klc[:, off:off + w].sum().item(); acc["mupw"][li] += (mu[..., off:off + w] ** 2).sum().item(); off += w
        mu_t, _ = m.encode(-x)                                  # time reversal: all spins n -> -n
        off = 0
        for li in range(4):
            w = 4 * (2 * li + 1)
            acc["tr"][li] += ((mu_t[..., off:off + w] - mu[..., off:off + w]) ** 2).sum().item(); acc["mu_norm"][li] += (mu[..., off:off + w] ** 2).sum().item(); off += w
        acc["n"] += B
    n = acc["n"]; A = acc["a"] / n; P = acc["pw"] / n
    a2, a4, a6 = A[2], A[4], A[6]
    odd = P[1] + P[3] + P[5]; even = P[2] + P[4] + P[6]
    mp = acc["mupw"] / acc["mupw"].sum()
    out = dict(run=os.path.basename(path)[:-3], T=a["T"], st=a["st"], R=a["R"], beta=a["beta"], alpha=a["alpha"],
               S=Sord.mean().item(), a2=a2, a4=a4, a6=a6, odd_even=odd / even,
               sigma2=(-(35 / 18) * a4 / a2 ** 2) if a2 > 0.3 else None,
               nll={str(L): acc["nll"][L] / n for L in Lcuts}, kl=acc["kl"] / n, kl_per_l=(acc["klch"] / n).tolist(),
               latent_power_frac=mp.tolist(), time_reversal_rel=(acc["tr"] / np.maximum(acc["mu_norm"], 1e-12)).tolist(),
               lam_power=P[1:].tolist())
    return out

if __name__ == "__main__":
    runs = sorted(glob.glob(f"{sys.argv[1]}/*.pt")); res = []
    for r in runs:
        res.append(evaluate(r, (1, 2, 4, 6))); print(res[-1]["run"], f"a2={res[-1]['a2']:.3f} a4={res[-1]['a4']:+.3f} odd/even={res[-1]['odd_even']:.1e} nll6={res[-1]['nll']['6']:.4f}", flush=True)
    json.dump(res, open(sys.argv[2], "w"), indent=1)
