import json, numpy as np, collections
def load(T):
    d = collections.defaultdict(list)
    for l in open(f"/home/user/S2_VAE_for_spin_structures/results/response/C_aniso_T{T}.jsonl"): r = json.loads(l); d[r["D"]].append(r)
    return {D: dict(n=len(v), **{k: (float(np.mean([x[k] for x in v])), float(np.std([x[k] for x in v], ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0) for k in "SAB"}) for D, v in sorted(d.items())}
if __name__ == "__main__":
    for T in (1.05, 0.95):
        try: r = load(T)
        except FileNotFoundError: continue
        print(T)
        for D, v in r.items(): print("  D=%+.3f n=%d S=%.3f±%.3f A=%.3f±%.3f B=%.3f±%.3f" % (D, v["n"], *v["S"], *v["A"], *v["B"]))
