import numpy as np, glob, re, collections
from defect_check import cells
def tri_energy_cells(S,b=3):
    A,B,C=S[:,:,0],S[:,:,1],S[:,:,2]
    up=A+B+C; dn=A+np.roll(B,1,axis=0)+np.roll(C,1,axis=1)
    e=(np.sum(up**2,-1)+np.sum(dn**2,-1))/2          # per unit cell, |S_tri|^2 averaged over the 2 triangles
    L=S.shape[0]; return e.reshape(L//b,b,L//b,b).mean(axis=(1,3)).ravel()
R=collections.defaultdict(list)
for f in glob.glob('seeds/kag_L12_T*_s*.npy'):
    T=float(re.search(r'T([\d.]+)_s',f).group(1))
    if T>0.0021: continue
    for S in np.load(f).astype(float)[::4]:
        z,p,A=cells(S,3); e=tri_energy_cells(S,3); eps2=np.mean(np.arcsin(np.clip(z,-1,1))**2,1)
        R[T]+=list(zip(A,e,eps2))
print(" T      | |psi_cell| bin | frac | <|S_tri|^2>  (thermal equipartition scale ~ T)  | <eps^2> in cell")
for T in sorted(R):
    a=np.array(R[T])
    for lo,hi in [(0.95,1.01),(0.85,0.95),(0.6,0.85),(0,0.6)]:
        s=(a[:,0]>=lo)&(a[:,0]<hi)
        if s.sum()<10: continue
        print(f" {T:.4f} | [{lo:.2f},{hi:.2f})   | {s.mean():.2f} | {a[s,1].mean():.5f}  ({a[s,1].mean()/T:.2f} T)                   | {a[s,2].mean():.3f}")
