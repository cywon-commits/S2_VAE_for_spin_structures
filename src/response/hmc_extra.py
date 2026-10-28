import json, sys
from run_B_mcmc import *
run, hs, nw, ns, B = sys.argv[1], [float(x) for x in sys.argv[2].split(",")], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
m = load_kag(f"{ROOT}/results/kag/{run}"); gen = torch.Generator().manual_seed(11); fn = f"{ROOT}/results/response/Bmc_{run[:-3]}.json"; out = json.load(open(fn))
for h in hs:
    r = hmc(m, h, nw, ns, B, gen); r["run"] = run; out.append(r); json.dump(out, open(fn, "w"), indent=1); print(h, r["m"], flush=True)
