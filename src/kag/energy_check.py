"""Data check: energy per site vs E = -1 + 11/12 T, seed count, seed-level spread.  python3 energy_check.py T ..."""
import sys, glob, numpy as np
sys.path.insert(0, ".."); sys.path.insert(0, "../kagome")
from kagome_mc import energy
for T in map(float, sys.argv[1:]):
    fs = sorted(glob.glob(f"../../data_kag/seeds/kag_L12_T{T:.4f}_s*.npy")); Es = []
    for f in fs:
        S = np.load(f).astype(float); Es.append(np.mean([energy(x) for x in S[::5]]))
    Es = np.array(Es); print(f"T={T}: {len(fs)} seeds, E/site = {Es.mean():.5f} +- {Es.std()/np.sqrt(len(Es)):.5f} (s.e.m.), expected {-1+11/12*T:.5f}, seed-to-seed std {Es.std():.5f}")
