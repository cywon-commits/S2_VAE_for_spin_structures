"""Oracle table on the full data (director-frame multipole fits), saved as JSON.  python3 oracle_table.py ../data ../results/oracle.json [val]"""
import sys, glob, json, numpy as np
from fitlib import Fitter
def frames(S):
    Q = np.einsum('cxyzi,cxyzj->cij', S, S) / S[0, ..., 0].size - np.eye(3) / 3
    w, V = np.linalg.eigh(Q); return 1.5 * w[:, -1], V[:, :, [1, 0, 2]]
rows = []
for f in sorted(glob.glob(f'{sys.argv[1]}/ll_L12_T*.npy')):
    S = np.load(f).astype(float); T = float(f.split('_T')[1][:5]); tag = f.split('_')[-1][0]
    if len(sys.argv) > 3: S = S[-max(len(S) // 10, 10):]          # same validation split as the VAE
    Sv, V = frames(S); Sm = Sv.mean(); Sp = np.einsum('cxyzi,cij->cxyzj', S, V).reshape(-1, 3)
    fits = {L: Fitter(L, order=59, alpha=1e-8).fit(Sp) for L in (1, 2, 4, 6)}
    f6 = fits[6]; P = f6.power(); a = {l: f6.coef(l, 0) * np.sqrt((2 * l + 1) / (4 * np.pi)) for l in (2, 4, 6)}
    odd = sum(P[l] for l in P if l % 2); even = sum(P[l] for l in P if l % 2 == 0)
    rows.append(dict(T=T, st=tag, S=Sm, a2=a[2], a4=a[4], a6=a[6], odd_even=odd / even, a2_MF=6 * Sm / T,
                     nll={str(L): fits[L].nll for L in fits}, dNLL={str(L): fits[L].nll - f6.nll for L in (1, 2, 4)},
                     sigma2=(-(35 / 18) * a[4] / a[2] ** 2) if a[2] > 0.3 else None))
    print(rows[-1], flush=True)
json.dump(rows, open(sys.argv[2], 'w'), indent=1)
