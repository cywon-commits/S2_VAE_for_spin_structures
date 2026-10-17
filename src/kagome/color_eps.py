import numpy as np, glob
from defect_check import cells
out={'ord':[], 'dis':[]}
for f in glob.glob('seeds/kag_L12_T0.0005_s*.npy')+glob.glob('seeds/kag_L12_T0.0010_s*.npy'):
    for S in np.load(f).astype(float)[::2]:
        z,p,A=cells(S,3); col=np.round(3*p/(2*np.pi)).astype(int)%3; eps2=np.arcsin(np.clip(z,-1,1))**2
        for c in range(len(A)):
            per=sorted([eps2[c][col[c]==k].mean() if np.any(col[c]==k) else np.nan for k in range(3)])
            out['ord' if A[c]>=0.95 else 'dis'].append(per)
for k,v in out.items():
    v=np.array(v); v=v[~np.isnan(v).any(1)]
    print(f"{'ordered (|psi|>=0.95)' if k=='ord' else 'disordered (|psi|<0.95)':26s} n={len(v):5d}  per-color <eps^2> sorted (min, mid, max): {v[:,0].mean():.3f}  {v[:,1].mean():.3f}  {v[:,2].mean():.3f}   ratio min/max = {np.mean(v[:,0]/v[:,2]):.2f}")
