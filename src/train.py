"""Train the SO(3)-equivariant multipole VAE.  python3 train.py --T 1.05 --st o --R 1 --beta 1 --steps 1500 --out ../results/runs"""
import argparse, json, os, time, numpy as np, torch
from mvae import MVAE
p = argparse.ArgumentParser()
p.add_argument("--T", type=float); p.add_argument("--st", default="o"); p.add_argument("--R", type=int, default=1)
p.add_argument("--beta", type=float, default=1.0); p.add_argument("--alpha", type=float, default=1e-6)
p.add_argument("--steps", type=int, default=1500); p.add_argument("--bs", type=int, default=8)
p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--seed", type=int, default=0)
p.add_argument("--nsub", type=int, default=128); p.add_argument("--data", default="../data"); p.add_argument("--out", default="../results/runs"); p.add_argument("--tag", default="")
a = p.parse_args()
torch.set_num_threads(1); torch.manual_seed(a.seed); np.random.seed(a.seed)
S = torch.tensor(np.load(f"{a.data}/ll_L12_T{a.T:.3f}_{a.st}.npy"))
nv = max(len(S) // 10, 10); tr, va = S[:-nv], S[-nv:]
m = MVAE(L=12, R=a.R); opt = torch.optim.Adam(m.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
name = f"T{a.T:.2f}{a.st}_R{a.R}_b{a.beta:g}_a{a.alpha:g}{a.tag}"; os.makedirs(a.out, exist_ok=True)
def val():
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(va), 10):
            o = m(va[i:i + 10], beta=a.beta, alpha=0, sample=False); out.append((o["nll"].item(), o["kl"].item()))
    m.train(); return np.mean(out, 0)
t0 = time.time(); log = []
for step in range(a.steps):
    idx = torch.randint(len(tr), (a.bs,)); x = tr[idx]
    # random global SO(3)/Z2 gauge augmentation is unnecessary (equivariant); random per-site sign flip is a gauge of the P2 model
    sg = torch.randint(0, 2, x.shape[:-1] + (1,)).float() * 2 - 1; x = x * sg
    beta = a.beta * min(1.0, step / (0.3 * a.steps))   # KL warm-up
    o = m(x, beta=beta, alpha=a.alpha, n_sub=a.nsub); opt.zero_grad(); o["loss"].backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sched.step()
    if step % 100 == 0 or step == a.steps - 1:
        v = val(); log.append((step, o["nll"].item(), o["kl"].item(), float(v[0]), float(v[1])))
        print(f"{name} step {step} train nll {o['nll'].item():.4f} kl {o['kl'].item():.2f} | val nll {v[0]:.4f} kl {v[1]:.2f} | {time.time()-t0:.0f}s", flush=True)
torch.save(dict(state=m.state_dict(), args=vars(a), log=log), f"{a.out}/{name}.pt")
json.dump(dict(args=vars(a), log=log, val_nll=log[-1][3], val_kl=log[-1][4]), open(f"{a.out}/{name}.json", "w"))
