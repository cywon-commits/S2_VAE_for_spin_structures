"""Oracle-latent analysis: latent = global director frame (Q-tensor eigvecs). Fit unrestricted exponential family
to site spins in that frame. Tests E1 predictions without training a VAE."""
import numpy as np, glob, sys
from fitlib import Fitter
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
xg,wg=leggauss(200); P2=lambda x:1.5*x**2-.5
def Sdual(a): e=np.exp(a*P2(xg)); return np.sum(wg*P2(xg)*e)/np.sum(wg*e)
def a2_of_S(S): return brentq(lambda a:Sdual(a)-S,-20,60) if S>0 else 0.0
def frames(S):
    Q=np.einsum('cxyzi,cxyzj->cij',S,S)/S[0,...,0].size-np.eye(3)/3
    w,V=np.linalg.eigh(Q); return 1.5*w[:,-1], V[:,:,[1,0,2]]   # columns: x=2nd, y=3rd, z=director
def local_spread(S,V,b=2):
    # director of b^3 blocks vs global director: <theta^2>
    c,L=S.shape[0],S.shape[1]; Sb=S.reshape(c,L//b,b,L//b,b,L//b,b,3)
    Qb=np.einsum('cxaybzdi,cxaybzdj->cxyzij',Sb,Sb)
    _,vb=np.linalg.eigh(Qb); db=vb[...,-1]; dg=V[:,:,2]
    cos=np.abs(np.einsum('cxyzi,ci->cxyz',db,dg)); return np.mean(np.arccos(np.clip(cos,0,1))**2)
rows=[]
for f in sorted(glob.glob('ll_L12_T*.npy')):
    S=np.load(f).astype(float); T=float(f.split('_T')[1][:5]); tag=f.split('_')[-1][0]
    Sv,V=frames(S); Sm=Sv.mean()
    Sp=np.einsum('cxyzi,cij->cxyzj',S,V).reshape(-1,3)
    fits={L:Fitter(L,order=59,alpha=1e-8).fit(Sp) for L in (1,2,4,6)}
    f6=fits[6]; P=f6.power()
    a={l:f6.coef(l,0)*np.sqrt((2*l+1)/(4*np.pi)) for l in (2,4,6)}
    odd=sum(P[l] for l in P if l%2); even=sum(P[l] for l in P if l%2==0)
    biax=f6.coef(2,2)**2+f6.coef(2,-2)**2
    dn={L:fits[L].nll-f6.nll for L in (1,2,4)}
    sig_est=-(35/18)*a[4]/a[2]**2 if a[2]>0.3 else np.nan
    rows.append((T,tag,Sm,a[2],a2_of_S(Sm),6*Sm/T,a[4],a[6],odd/even,biax,dn,sig_est,local_spread(S,V,2),local_spread(S,V,3) if 12%3==0 else np.nan))
print(" T    st |  S    | a2_fit a2*(S) a2_MF | a4      a6     | odd/even  biax(m=2) | dNLL L1,L2,L4          | sig2_est  loc2^3 loc3^3")
for r in rows:
    T,tag,Sm,a2,a2s,a2mf,a4,a6,oe,bx,dn,se,l2,l3=r
    print(f"{T:.2f} {tag} | {Sm:.3f} | {a2:6.3f} {a2s:6.3f} {a2mf:6.3f} | {a4:+.4f} {a6:+.4f} | {oe:.1e}  {bx:.1e}  | {dn[1]:.3f} {dn[2]:.4f} {dn[4]:.4f} | {se:+.4f}  {l2:.3f}  {l3:.3f}")
