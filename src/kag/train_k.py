"""python3 train_k.py --T 0.0005 --R 1 --steps 20000 --bs 16 --out ../../results/kag"""
import argparse, json, os, time, numpy as np, torch
from kmodel import KVAE, kag_features, kag_stats
from kmodel5 import KVAE5
from kdata import load_T, TRAIN_SEEDS, VAL_SEEDS
p = argparse.ArgumentParser()
p.add_argument("--T", type=float); p.add_argument("--R", type=int, default=1); p.add_argument("--steps", type=int, default=20000); p.add_argument("--bs", type=int, default=16)
p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--alpha", type=float, default=1e-8); p.add_argument("--beta", type=float, default=1.0); p.add_argument("--kl_warm", type=float, default=0.3)
p.add_argument("--warm", default=""); p.add_argument("--prune", action="store_true"); p.add_argument("--seed", type=int, default=0); p.add_argument("--order", type=int, default=59); p.add_argument("--up", action="store_true"); p.add_argument("--tau0", type=float, default=0.5)
p.add_argument("--out", default="../../results/kag"); p.add_argument("--tag", default=""); p.add_argument("--threads", type=int, default=1)
a = p.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
def prep(S):                                         # original and globally time-reversed copies
    S = torch.tensor(S); R = a.R
    return torch.cat([kag_features(S, R), kag_features(-S, R)]), torch.cat([kag_stats(S, R), kag_stats(-S, R)]), len(S)
Ftr, Str, ntr = prep(load_T(a.T, TRAIN_SEEDS)); Fva, Sva, nva = prep(load_T(a.T, VAL_SEEDS)); nsub = (12 // a.R) ** 2
m = KVAE5(R=a.R, tau0=a.tau0, order=a.order) if a.up else KVAE(R=a.R, order=a.order)
if a.warm: m.load_state_dict({k: v for k, v in torch.load(a.warm, weights_only=False)["state"].items() if not k.startswith("logz.")}, strict=False)
if a.prune:
    with torch.no_grad():
        mu, lv = m.encode(Ftr[:2000]); lvx = m.expand_logvar(lv); klc = (0.5 * (mu ** 2 + lvx.exp() - 1 - lvx)).mean((0, 1)); mask = torch.zeros_like(klc); off = 0
        for l in range(4):
            for c in range(4):
                w = 2 * l + 1
                if klc[off:off + w].sum() > 0.5: mask[off:off + w] = 1
                off += w
        m.zmask.copy_(mask); print("pruned: active components", int(mask.sum()), flush=True)
opt = torch.optim.Adam(m.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05, final_div_factor=1e3)
name = f"{'kagu' if a.up else 'kag'}_T{a.T:.4f}_R{a.R}{a.tag}"; os.makedirs(a.out, exist_ok=True); ck = f"{a.out}/{name}.ckpt"; step0, log = 0, []
if os.path.exists(ck):
    c = torch.load(ck, weights_only=False); m.load_state_dict(c["state"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"]); step0, log = c["step"], c["log"]; print("resumed at", step0, flush=True)
def val():
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, nva, 40):
            idx = torch.arange(i, min(i + 40, nva)); o = m.forward_stats(Fva[idx], Sva[idx], nsub); out.append((o["nll"].item(), o["kl"].item()))
    m.train(); return np.mean(out, 0)
t0 = time.time()
for step in range(step0, a.steps):
    idx = torch.randint(0, 2 * ntr, (a.bs,)); beta = a.beta * min(1.0, step / max(1, a.kl_warm * a.steps))
    o = m.forward_stats(Ftr[idx], Str[idx], nsub, beta=beta, alpha=a.alpha); opt.zero_grad(); o["loss"].backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sched.step()
    if step % 250 == 0 or step == a.steps - 1:
        v = val(); log.append((step, o["nll"].item(), o["kl"].item(), float(v[0]), float(v[1])))
        print(f"{name} step {step} train nll {o['nll'].item():.4f} kl {o['kl'].item():.1f} | val nll {v[0]:.4f} kl {v[1]:.1f} | {time.time()-t0:.0f}s", flush=True)
    if step % 250 == 249 or step == a.steps - 1:
        torch.save(dict(state=m.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step + 1, log=log, args=vars(a)), ck + ".tmp"); os.replace(ck + ".tmp", ck)
torch.save(dict(state=m.state_dict(), args=vars(a), log=log), f"{a.out}/{name}.pt")
