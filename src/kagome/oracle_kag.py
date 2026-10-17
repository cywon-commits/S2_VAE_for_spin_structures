import numpy as np, glob, re
def analyze(S):
    c=S.shape[0]; X=S.reshape(c,-1,3).astype(float); N=X.shape[1]
    Q=np.einsum('cni,cnj->cij',X,X)/N; w,V=np.linalg.eigh(Q)
    nu=V[:,:,0]; e1=V[:,:,2]; e2=np.cross(nu,e1)
    lam=w[:,0]                                   # coplanarity: 0 = coplanar, 1/3 isotropic
    z=np.einsum('cni,ci->cn',X,nu); x=np.einsum('cni,ci->cn',X,e1); y=np.einsum('cni,ci->cn',X,e2)
    phi=np.arctan2(y,x); psi=np.mean(np.exp(3j*phi),1); phi0=np.angle(psi)/3
    p=phi-phi0[:,None]; eps=np.arcsin(np.clip(z,-1,1))
    r=dict(lam=lam.mean(),oct=np.abs(psi).mean(),
           m3=np.mean(np.cos(3*p)),m6=np.mean(np.cos(6*p)),m9=np.mean(np.cos(9*p)),
           s3=np.mean(np.sin(3*p)),c1=np.abs(np.mean(np.exp(1j*p))),c2=np.abs(np.mean(np.exp(2j*p))),
           eps2=np.mean(eps**2),P2p=np.mean(1.5*z**2-.5)+0.5)
    r['sig_par2']=-(2/9)*np.log(r['m3']); return r
rows=[]
for f in sorted(glob.glob('kag_L*_T*.npy')):
    L=int(re.search(r'L(\d+)',f).group(1)); T=float(re.search(r'T([\d.]+)\.npy',f).group(1))
    rows.append((L,T,analyze(np.load(f))))
rows.sort()
print(" L   T      | lam_min  |psi3| | m3     m6     m3^4   m9    m3^9 | <sin3> |e^i1| |e^i2| | sig_par2  <eps^2>  (3/2)<eps2>vs<P2>+.5")
for L,T,r in rows:
    print(f"{L:3d} {T:.4f} | {r['lam']:.4f} {r['oct']:.3f} | {r['m3']:.4f} {r['m6']:.4f} {r['m3']**4:.4f} {r['m9']:.4f} {r['m3']**9:.4f} | {r['s3']:+.4f} {r['c1']:.4f} {r['c2']:.4f} | {r['sig_par2']:.5f}  {r['eps2']:.5f}  {1.5*r['eps2']:.4f} vs {r['P2p']:.4f}")
np.save('kag_rows.npy',np.array([(L,T,r['sig_par2'],r['eps2'],r['m3'],r['m6'],r['lam'],r['oct']) for L,T,r in rows]))
