"""v4 task C: sigma^2 / a_loc from the DECODER-FRAME oracle: the latent only fixes each cell's axes (decoded l=2 principal frame at the posterior mean);
cell spins are rotated into those frames, pooled, and the moment estimator (a_loc, sigma2) and the unrestricted multipole fit are applied to the spins.
python3 frame_sigma.py out.json run.pt ...   (R=1 baseline: pass a v2/v3 R=1 checkpoint; MVAE5 for v4u runs)"""
import sys, json, numpy as np, torch
from scipy.special import eval_legendre as Pl
from mvae2 import MVAE2, cell_features, compute_stats
from mvae5 import MVAE5
from eval_A import frame_of
from fitlib import Fitter
from momlib import est_vec
torch.set_num_threads(1)
def load_any(path):
    ck = torch.load(path, weights_only=False); a = ck["args"]; cls = MVAE5 if a.get("up") else MVAE2
    m = cls(L=12, R=a["R"]); m.load_state_dict(ck["state"], strict=False); m.eval(); return m, a
@torch.no_grad()
def frame_sigma(path):
    m, a = load_any(path); R = a["R"]; b = 12 // R; nc = b ** 3
    S = torch.tensor(np.load(f"{a['data']}/ll_L12_T{a['T']:.3f}_{a['st']}.npy")); nv = max(len(S) // 10, 10); Sv = S[-nv:]; F = cell_features(Sv, R); St = compute_stats(Sv, R)
    mu, lv = (m.encode_c(F) if isinstance(m, MVAE5) else m.encode(feat=F)); lam = m.decode_cells(mu); V = frame_of(lam)
    nll_dec = ((-(lam * St).sum(-1) + nc * m.logz(lam)).sum(-1).mean().item()) / 1728
    Sb = Sv.double().reshape(nv, R, b, R, b, R, b, 3).permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(nv, R ** 3, nc, 3)
    Sp = torch.einsum("cknj,ckji->ckni", Sb, V).reshape(-1, 3).numpy(); c = np.abs(Sp[:, 2])
    aL, s2 = est_vec(np.array([Pl(2, c).mean()]), np.array([Pl(4, c).mean()])); fit = Fitter(6, order=59, alpha=1e-8).fit(Sp)
    return dict(run=path.split("/")[-1][:-3], T=a["T"], R=R, nll_decoder_meanz=nll_dec, nll_frame_oracle=fit.nll, a_loc=float(aL[0]), sigma2=float(s2[0]))
if __name__ == "__main__":
    out = [frame_sigma(p) for p in sys.argv[2:]]
    for r in out: print(r["run"], "decoder NLL %.4f frame-oracle NLL %.4f a_loc %.3f sigma2 %.4f" % (r["nll_decoder_meanz"], r["nll_frame_oracle"], r["a_loc"], r["sigma2"]), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
