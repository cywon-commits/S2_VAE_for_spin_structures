"""Data check vs the chat oracle table: global-frame moments over all 40 seeds (seed-level mean +- s.e.m. by bootstrap over seeds).  python3 data_check_k.py out.json T ..."""
import sys, json, numpy as np
from oracle_k import to_frame
from kdata import load_T
rng = np.random.default_rng(0); out = []
CHAT = {0.0005: (0.041, 2.42, 0.086), 0.001: (0.158, 1.73, 0.167), 0.002: (0.223, 2.07, 0.210), 0.004: (0.537, 2.02, 0.314)}
for T in map(float, sys.argv[2:]):
    per = []
    for s in range(1, 41):
        F = to_frame(load_T(T, [s])).reshape(-1, 3); z = np.clip(F[:, 2], -1, 1); phi = np.arctan2(F[:, 1], F[:, 0])
        per.append([np.mean(np.cos(3 * phi)), np.mean(np.cos(6 * phi)), np.mean(np.cos(9 * phi)), np.mean(np.arcsin(z) ** 2)])
    per = np.array(per)
    def stat(idx):
        M = per[idx].mean(0); return -np.log(M[0]), np.log(M[1]) / np.log(M[0]), np.log(M[2]) / np.log(M[0]), M[3]
    full = stat(np.arange(40)); boots = np.array([stat(rng.integers(0, 40, 40)) for _ in range(300)]); se = boots.std(0)
    # same with only the first 6 seeds, as in the chat table
    six = stat(np.arange(6))
    r = dict(T=T, minus_ln_m3=full[0], se_minus_ln_m3=se[0], ln_m6_ln_m3=full[1], se_m6=se[1], ln_m9_ln_m3=full[2], se_m9=se[2], eps2=full[3], se_eps2=se[3], first6=dict(minus_ln_m3=six[0], ln_m6_ln_m3=six[1], eps2=six[3]), chat=CHAT[T],
             seed_spread_m3=float(per[:, 0].std()))
    out.append(r); c = CHAT[T]
    print(f"T={T}: -ln m3 {full[0]:.3f}+-{se[0]:.3f} (chat {c[0]}, first-6-seeds {six[0]:.3f}) | ln m6/ln m3 {full[1]:.2f}+-{se[1]:.2f} (chat {c[1]}, first6 {six[1]:.2f}) | ln m9/ln m3 {full[2]:.2f} | <eps2> {full[3]:.3f}+-{se[3]:.3f} (chat {c[2]}, first6 {six[3]:.3f})", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
