"""v2 training with sufficient statistics, resumable.
python3 train2.py --T 1.05 --st o --R 1 --steps 30000 --bs 16 --out ../results/v2"""
import argparse, json, os, time, numpy as np, torch
from mvae2 import MVAE2, compute_stats, cell_features
from mvae3 import MVAE3
from mvae4 import MVAE4
from mvae5 import MVAE5
p = argparse.ArgumentParser()
p.add_argument("--T", type=float); p.add_argument("--st", default="o"); p.add_argument("--R", type=int, default=1)
p.add_argument("--beta", type=float, default=1.0); p.add_argument("--alpha", type=float, default=1e-5)
p.add_argument("--steps", type=int, default=30000); p.add_argument("--bs", type=int, default=16)
p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--seed", type=int, default=0)
p.add_argument("--warm", default=""); p.add_argument("--prune", action="store_true"); p.add_argument("--hier", action="store_true"); p.add_argument("--sub", action="store_true"); p.add_argument("--up", action="store_true"); p.add_argument("--tau0", type=float, default=0.5); p.add_argument("--kl_warm", type=float, default=0.3)
p.add_argument("--threads", type=int, default=1); p.add_argument("--data", default="../data"); p.add_argument("--out", default="../results/v2"); p.add_argument("--tag", default="")
a = p.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
S = torch.tensor(np.load(f"{a.data}/ll_L12_T{a.T:.3f}_{a.st}.npy")); nv = max(len(S) // 10, 10)
St = compute_stats(S, a.R); F = cell_features(S, a.R); ncell = (12 // a.R) ** 3
tr, va = slice(0, len(S) - nv), slice(len(S) - nv, len(S))
m = MVAE5(L=12, R=a.R, tau0=a.tau0) if a.up else (MVAE3 if a.hier else MVAE4 if a.sub else MVAE2)(L=12, R=a.R)
if a.warm: m.load_state_dict(torch.load(a.warm, weights_only=False)["state"], strict=False)
if a.sub:
    with torch.no_grad(): print('main active l=2 channel', m.set_active(F[:512]), flush=True)
if a.prune:
    with torch.no_grad(): mu_, lv_ = m.encode(feat=F[:len(S) - nv]); m.zmask.copy_(m.active_channels(mu_, lv_))
    print("pruned: active components", int(m.zmask.sum().item()), "of", m.zmask.numel(), flush=True)
opt = torch.optim.Adam(m.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05, final_div_factor=1e3)
name = f"{'v4u' if a.up else 'v3h' if a.hier else 'v3s' if a.sub else 'v2'}_T{a.T:.2f}{a.st}_R{a.R}_b{a.beta:g}{a.tag}"; os.makedirs(a.out, exist_ok=True); ck = f"{a.out}/{name}.ckpt"
step0, log = 0, []
if os.path.exists(ck):
    c = torch.load(ck, weights_only=False); m.load_state_dict(c["state"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"]); step0, log = c["step"], c["log"]
    print("resumed at", step0, flush=True)
def val():
    m.eval(); out = []
    with torch.no_grad():
        for i in range(len(S) - nv, len(S), 20):
            o = m.forward_stats(F[i:i + 20], St[i:i + 20], ncell, beta=1.0, alpha=0, sample=True); out.append((o["nll"].item(), o["kl"].item()))
    m.train(); return np.mean(out, 0)
t0 = time.time()
for step in range(step0, a.steps):
    idx = torch.randint(0, len(S) - nv, (a.bs,))
    beta = a.beta * min(1.0, step / max(1, a.kl_warm * a.steps)) if not a.warm else a.beta * min(1.0, step / max(1, a.kl_warm * a.steps))
    o = m.forward_stats(F[idx], St[idx], ncell, beta=beta, alpha=a.alpha)
    opt.zero_grad(); o["loss"].backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sched.step()
    if step % 250 == 0 or step == a.steps - 1:
        v = val(); log.append((step, o["nll"].item(), o["kl"].item(), float(v[0]), float(v[1])))
        print(f"{name} step {step} train nll {o['nll'].item():.4f} kl {o['kl'].item():.1f} | val nll {v[0]:.4f} kl {v[1]:.1f} | {time.time()-t0:.0f}s", flush=True)
    if step % 1000 == 999 or step == a.steps - 1:
        torch.save(dict(state=m.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step + 1, log=log, args=vars(a)), ck + ".tmp"); os.replace(ck + ".tmp", ck)
torch.save(dict(state=m.state_dict(), args=vars(a), log=log), f"{a.out}/{name}.pt")
