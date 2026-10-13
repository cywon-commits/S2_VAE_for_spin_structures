"""(f) decoder order-truncation ablation: freeze encoder, re-fit decoder (d1,d2,dlin) with l<=Lc for each Lc, measure val NLL.
python3 ablate.py run.pt out.json [steps]"""
import sys, json, copy, numpy as np, torch
from evaluate import load, val_data
torch.set_num_threads(1)
path, out = sys.argv[1], sys.argv[2]; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 300
m0, a = load(path); assert m0.R == 1, "ablation implemented for global latent (R=1)"
S = torch.tensor(np.load(f"{a['data']}/ll_L12_T{a['T']:.3f}_{a['st']}.npy")); nv = max(len(S) // 10, 10); tr, va = S[:-nv], S[-nv:]
res = {}
for Lc in (6, 4, 2, 1):
    torch.manual_seed(0); m = copy.deepcopy(m0)
    dec = [p for n_, p in m.named_parameters() if n_.startswith(("d1", "d2", "dlin"))]
    for p in m.parameters(): p.requires_grad_(False)
    for p in dec: p.requires_grad_(True)
    opt = torch.optim.Adam(dec, lr=2e-3)
    for step in range(steps):
        x = tr[torch.randint(len(tr), (4,))]
        with torch.no_grad(): mu, lv = m.encode(x)
        z = m.sample(mu, lv); lam = m.decode(z)
        loss = m.nll_sites(lam, x.reshape(4, -1, 3), Lc).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    tot = 0.; torch.manual_seed(1)
    with torch.no_grad():
        for i in range(0, len(va), 10):
            x = va[i:i + 10]; mu, lv = m.encode(x)
            for _ in range(4): tot += m.nll_sites(m.decode(m.sample(mu, lv)), x.reshape(len(x), -1, 3), Lc).mean(1).sum().item() / 4
    res[str(Lc)] = tot / len(va); print(path, Lc, res[str(Lc)], flush=True)
res["dNLL"] = {L: res[L] - res["6"] for L in ("1", "2", "4")}
json.dump(dict(run=path.split("/")[-1][:-3], T=a["T"], st=a["st"], **res), open(out, "w"), indent=1)
