"""python3 train_k6u.py --T 0.0005 --mode oracle|shuffled|nolatent --steps 8000 --warm ../../results/kag/kagu_T0.0005_R4.pt
Same recipe as the v2 site-resolved model (encoder warm start from the v1 up/down run, decoder from scratch, 8000 steps).  mode nolatent: latent fixed to 0 (only u_{c_i} reaches the decoder)."""
import argparse, os, time, numpy as np, torch
from kmodel import kag_features
from kmodel6 import site_Y
from kmodel6u import KVAE6U
from ring_oracle import global_triad
from kdata import load_T, TRAIN_SEEDS, VAL_SEEDS
def oracle_u(S):
    """S [C,12,12,3,3] -> u_{c_i} [C,12,12,3,3]: the triad direction closest to each spin (global_triad of the configuration itself)."""
    out = np.zeros(S.shape, np.float32)
    for i in range(len(S)):
        X = S[i].reshape(-1, 3).astype(float); U = global_triad(X); out[i] = U[np.argmax(X @ U.T, 1)].reshape(S.shape[1:])
    return out
def site_vec(u, R):
    B, L = u.shape[:2]; b = L // R; return torch.as_tensor(u).reshape(B, R, b, R, b, 3, 3).permute(0, 1, 3, 2, 4, 5, 6).reshape(B, R * R, b * b * 3, 3)
def shuffle_sites(u, g):
    flat = torch.as_tensor(u).reshape(len(u), -1, 3).clone()
    for i in range(len(flat)): flat[i] = flat[i][torch.randperm(flat.shape[1], generator=g)]
    return flat.reshape(u.shape)
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=float); p.add_argument("--R", type=int, default=4); p.add_argument("--steps", type=int, default=8000); p.add_argument("--bs", type=int, default=16); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=1e-8); p.add_argument("--kl_warm", type=float, default=0.05); p.add_argument("--warm", default=""); p.add_argument("--mode", default="oracle")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default="../../results/kag"); p.add_argument("--tag", default="")
    a = p.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed); np.random.seed(a.seed); R = a.R
    Str = torch.tensor(load_T(a.T, TRAIN_SEEDS)); Sva = torch.tensor(load_T(a.T, VAL_SEEDS)); ntr, nva = len(Str), len(Sva)
    Ftr = torch.cat([kag_features(Str, R), kag_features(-Str, R)]); Fva = kag_features(Sva, R)
    Utr = torch.cat([torch.as_tensor(oracle_u(Str.numpy())), torch.as_tensor(oracle_u(-Str.numpy()))]); Uva = torch.as_tensor(oracle_u(Sva.numpy()))
    if a.mode == "shuffled": g = torch.Generator().manual_seed(123); Utr = shuffle_sites(Utr, g); Uva = shuffle_sites(Uva, g)
    m = KVAE6U(R=R)
    if a.warm and a.mode != "nolatent":
        sd = {k: v for k, v in torch.load(a.warm, weights_only=False)["state"].items() if k.split(".")[0] in ("c1", "c2", "mu", "lv", "log_tau", "zmask")}; print(m.load_state_dict(sd, strict=False).unexpected_keys, flush=True)
    if a.mode == "nolatent": m.zmask.zero_()
    opt = torch.optim.Adam(m.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05, final_div_factor=1e3)
    name = f"kag6u_T{a.T:.4f}_R{R}_{a.mode}{a.tag}"; os.makedirs(a.out, exist_ok=True); ck = f"{a.out}/{name}.ckpt"; step0, log = 0, []
    if os.path.exists(ck):
        c = torch.load(ck, weights_only=False); m.load_state_dict(c["state"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"]); step0, log = c["step"], c["log"]; print("resumed at", step0, flush=True)
    def batch(idx):
        tr = idx >= ntr; i = idx % ntr; Y = torch.cat([site_Y(Str[i[g]], R, sg) for g, sg in ((~tr, 1.0), (tr, -1.0)) if g.any()]); F = torch.cat([Ftr[idx[~tr]], Ftr[idx[tr]]]); U = torch.cat([Utr[idx[~tr]], Utr[idx[tr]]]); return F, Y, site_vec(U, R)
    def val():
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, nva, 20):
                sl = slice(i, min(i + 20, nva)); o = m.forward_stats(Fva[sl], site_Y(Sva[sl], R), site_vec(Uva[sl], R)); out.append((o["nll"].item(), o["kl"].item()))
        m.train(); return np.mean(out, 0)
    t0 = time.time()
    for step in range(step0, a.steps):
        idx = torch.randint(0, 2 * ntr, (a.bs,)); beta = min(1.0, step / max(1, a.kl_warm * a.steps)); F, Y, U = batch(idx)
        o = m.forward_stats(F, Y, U, beta=beta, alpha=a.alpha); opt.zero_grad(); o["loss"].backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sched.step()
        if step % 250 == 0 or step == a.steps - 1:
            v = val(); log.append((step, o["nll"].item(), o["kl"].item(), float(v[0]), float(v[1])))
            print(f"{name} step {step} train nll {o['nll'].item():.4f} kl {o['kl'].item():.1f} | val nll {v[0]:.4f} kl {v[1]:.1f} | {time.time()-t0:.0f}s", flush=True)
        if step % 250 == 249 or step == a.steps - 1:
            torch.save(dict(state=m.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step + 1, log=log, args=vars(a)), ck + ".tmp"); os.replace(ck + ".tmp", ck)
    torch.save(dict(state=m.state_dict(), args=dict(vars(a), up=True, dec="site+cap"), log=log), f"{a.out}/{name}.pt")
