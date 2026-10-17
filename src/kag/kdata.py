"""Seed-level split of the kagome data.  Seeds 1..32 train, 33..40 validation (a seed's snapshots never appear on both sides)."""
import glob, re, numpy as np, torch
TRAIN_SEEDS = range(1, 33); VAL_SEEDS = range(33, 41)
def load_T(T, seeds, root="../../data_kag/seeds"):
    out = []
    for s in seeds:
        f = f"{root}/kag_L12_T{T:.4f}_s{s}.npy"; out.append(np.load(f))
    return np.concatenate(out)            # [n_seeds*100, 12, 12, 3, 3]
