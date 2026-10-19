"""python3 train_k6c.py --T 0.0005 --R 4 --color oracle|shuffled --steps 6000 --warm ../../results/kag/kag6_T0.0005_R4.pt   (fine-tune of the v2 site-resolved model with colour input)"""
import argparse, os, time, numpy as np, torch
from kmodel import kag_features
from kmodel6 import site_Y
from kmodel6c import KVAE6C, oracle_colors, site_col
from kdata import load_T, TRAIN_SEEDS, VAL_SEEDS
p = argparse.ArgumentParser()
p.add_argument("--T", type=float); p.add_argument("--R", type=int, default=4); p.add_argument("--steps", type=int, default=6000); p.add_argument("--bs", type=int, default=16); p.add_argument("--lr", type=float, default=7e-4)
p.add_argument("--alpha", type=float, default=1e-8); p.add_argument("--kl_warm", type=float, default=0.0); p.add_argument("--warm", default=""); p.add_argument("--color", default="oracle")
p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default="../../results/kag"); p.add_argument("--tag", default="")
a = p.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed); np.random.seed(a.seed); R = a.R
Str = torch.tensor(load_T(a.T, TRAIN_SEEDS)); Sva = torch.tensor(load_T(a.T, VAL_SEEDS)); ntr, nva = len(Str), len(Sva)
Ftr = torch.cat([kag_features(Str, R), kag_features(-Str, R)]); Fva = kag_features(Sva, R)
Ctr = torch.cat([torch.as_tensor(oracle_colors(Str.numpy())), torch.as_tensor(oracle_colors(-Str.numpy()))]); Cva = torch.as_tensor(oracle_colors(Sva.numpy()))
def shuffle(C, g):                                    # random permutation of the colour labels over the sites of each configuration
    flat = C.reshape(len(C), -1).clone()
    for i in range(len(flat)): flat[i] = flat[i][torch.randperm(flat.shape[1], generator=g)]
    return flat.reshape(C.shape)
if a.color == "shuffled": g = torch.Generator().manual_seed(123); Ctr = shuffle(Ctr, g); Cva = shuffle(Cva, g)
m = KVAE6C(R=R)
if a.warm: print(m.load_state_dict({k: v for k, v in torch.load(a.warm, weights_only=False)["state"].items() if not k.startswith("logz.")}, strict=False), flush=True)
opt = torch.optim.Adam(m.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05, final_div_factor=1e3)
name = f"kag6c_T{a.T:.4f}_R{R}_{a.color}{a.tag}"; os.makedirs(a.out, exist_ok=True); ck = f"{a.out}/{name}.ckpt"; step0, log = 0, []
if os.path.exists(ck):
    c = torch.load(ck, weights_only=False); m.load_state_dict(c["state"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"]); step0, log = c["step"], c["log"]; print("resumed at", step0, flush=True)
def batch(idx):
    tr = idx >= ntr; i = idx % ntr; Y = torch.cat([site_Y(Str[i[g]], R, sg) for g, sg in ((~tr, 1.0), (tr, -1.0)) if g.any()]); F = torch.cat([Ftr[idx[~tr]], Ftr[idx[tr]]]); C = torch.cat([Ctr[idx[~tr]], Ctr[idx[tr]]]); return F, Y, site_col(C, R)
def val():
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, nva, 20):
            sl = slice(i, min(i + 20, nva)); o = m.forward_stats(Fva[sl], site_Y(Sva[sl], R), site_col(Cva[sl], R)); out.append((o["nll"].item(), o["kl"].item()))
    m.train(); return np.mean(out, 0)
t0 = time.time()
for step in range(step0, a.steps):
    idx = torch.randint(0, 2 * ntr, (a.bs,)); beta = min(1.0, step / max(1, a.kl_warm * a.steps)) if a.kl_warm > 0 else 1.0; F, Y, C = batch(idx)
    o = m.forward_stats(F, Y, C, beta=beta, alpha=a.alpha); opt.zero_grad(); o["loss"].backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sched.step()
    if step % 250 == 0 or step == a.steps - 1:
        v = val(); log.append((step, o["nll"].item(), o["kl"].item(), float(v[0]), float(v[1])))
        print(f"{name} step {step} train nll {o['nll'].item():.4f} kl {o['kl'].item():.1f} | val nll {v[0]:.4f} kl {v[1]:.1f} | {time.time()-t0:.0f}s", flush=True)
    if step % 250 == 249 or step == a.steps - 1:
        torch.save(dict(state=m.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step + 1, log=log, args=vars(a)), ck + ".tmp"); os.replace(ck + ".tmp", ck)
torch.save(dict(state=m.state_dict(), args=dict(vars(a), up=True, dec="site+colour"), log=log), f"{a.out}/{name}.pt")
