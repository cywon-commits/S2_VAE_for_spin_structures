"""classify_blocks (kagome/block_defects.py rule) returning the defect count of one configuration."""
import sys, numpy as np
from style import ROOT
sys.path.insert(0, f"{ROOT}/src/kag"); sys.path.insert(0, f"{ROOT}/src/kagome")
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/src/kagome")
from cfg_terms import block_frames
def classify_blocks_count(S, b):
    C = block_frames(S, b); r = lambda X, dx, dy: np.roll(X, (-dx, -dy), axis=(0, 1)); loop = [C, r(C, 1, 0), r(C, 1, 1), r(C, 0, 1), C]; q = C[:, :, 0, :]
    for X in loop[1:]: idx = np.argmax(np.einsum('xyci,xyi->xyc', X, q), -1); q = np.take_along_axis(X, idx[..., None, None], 2)[:, :, 0, :]
    k = idx // 4; sg = (idx // 2) % 2; mn = idx % 2; none = (k == 0) & (sg == 0) & (mn == 0); return int((~none).sum())
