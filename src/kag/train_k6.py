"""python3 train_k6.py --T 0.0005 --R 4 --steps 8000 --bs 16 --warm ../../results/kag/kagu_T0.0005_R4.pt   (encoder warm start from the v1 up/down run; decoder from scratch)"""
import argparse, os, time, numpy as np, torch
from kmodel import kag_features
from kmodel6 import KVAE6, site_Y
from kdata import load_T, TRAIN_SEEDS, VAL_SEEDS
p = argparse.ArgumentParser()
p.add_argument("--T", type=float); p.add_argument("--R", type=int, default=4); p.add_argument("--steps", type=int, default=8000); p.add_argument("--bs", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--alpha", type=float, default=1e-8); p.add_argument("--kl_warm", type=float, default=0.05); p.add_argument("--warm", default="")
p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default="../../results/kag"); p.add_argument("--tag", default=""); p.add_argument("--tau0", type=float, default=0.5)
a = p.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed); np.random.seed(a.seed); R = a.R
Str = torch.tensor(load_T(a.T, TRAIN_SEEDS)); Sva = torch.tensor(load_T(a.T, VAL_SEEDS)); ntr, nva = len(Str), len(Sva)
Ftr = torch.cat([kag_features(Str, R), kag_features(-Str, R)]); Fva = kag_features(Sva, R)
m = KVAE6(R=R, tau0=a.tau0)
if a.warm:
    sd = {k: v for k, v in torch.load(a.warm, weights_only=False)["state"].items() if k.split(".")[0] in ("c1", "c2", "mu", "lv", "log_tau", "zmask")}; print(m.load_state_dict(sd, strict=False).unexpected_keys, flush=True)
opt = torch.optim.Adam(m.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05, final_div_factor=1e3)
name = f"kag6_T{a.T:.4f}_R{R}{a.tag}"; os.makedirs(a.out, exist_ok=True); ck = f"{a.out}/{name}.ckpt"; step0, log = 0, []
if os.path.exists(ck):
    c = torch.load(ck, weights_only=False); m.load_state_dict(c["state"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"]); step0, log = c["step"], c["log"]; print("resumed at", step0, flush=True)
def batch(idx):                                   # idx in [0, 2*ntr): second half = time-reversed copies
    tr = idx >= ntr; i = idx % ntr; Y = torch.cat([site_Y(Str[i[g]], R, sg) for g, sg in ((~tr, 1.0), (tr, -1.0)) if g.any()]); F = torch.cat([Ftr[idx[~tr]], Ftr[idx[tr]]]); return F, Y
def val():
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, nva, 20):
            sl = slice(i, min(i + 20, nva)); o = m.forward_stats(Fva[sl], site_Y(Sva[sl], R)); out.append((o["nll"].item(), o["kl"].item()))
    m.train(); return np.mean(out, 0)
t0 = time.time()
for step in range(step0, a.steps):
    idx = torch.randint(0, 2 * ntr, (a.bs,)); beta = min(1.0, step / max(1, a.kl_warm * a.steps)); F, Y = batch(idx)
    o = m.forward_stats(F, Y, beta=beta, alpha=a.alpha); opt.zero_grad(); o["loss"].backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sched.step()
    if step % 250 == 0 or step == a.steps - 1:
        v = val(); log.append((step, o["nll"].item(), o["kl"].item(), float(v[0]), float(v[1])))
        print(f"{name} step {step} train nll {o['nll'].item():.4f} kl {o['kl'].item():.1f} | val nll {v[0]:.4f} kl {v[1]:.1f} | {time.time()-t0:.0f}s", flush=True)
    if step % 250 == 249 or step == a.steps - 1:
        torch.save(dict(state=m.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step + 1, log=log, args=vars(a)), ck + ".tmp"); os.replace(ck + ".tmp", ck)
torch.save(dict(state=m.state_dict(), args=dict(vars(a), up=True, dec="site"), log=log), f"{a.out}/{name}.pt")
