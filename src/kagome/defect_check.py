import numpy as np, glob, re, collections
def cells(S,b):
    L=S.shape[0]; c=S.reshape(L//b,b,L//b,b,3,3).transpose(0,2,1,3,4,5).reshape(-1,3*b*b,3)
    Q=np.einsum('cni,cnj->cij',c,c); V=np.linalg.eigh(Q)[1]; nu=V[:,:,0]; e1=V[:,:,2]; e2=np.cross(nu,e1)
    z=np.einsum('cni,ci->cn',c,nu); phi=np.arctan2(np.einsum('cni,ci->cn',c,e2),np.einsum('cni,ci->cn',c,e1))
    psi=np.mean(np.exp(3j*phi),1); return z,phi-np.angle(psi)[:,None]/3,np.abs(psi)
D=collections.defaultdict(list)
for f in glob.glob('seeds/kag_L12_T*_s*.npy'):
    T=float(re.search(r'T([\d.]+)_s',f).group(1))
    if T>0.0041: continue
    for S in np.load(f).astype(float)[::4]: D[T].append(cells(S,3))
print("cells of 27 sites, split by local octupolar order |psi_cell|")
print(" T      | bin |psi|      frac  | <eps^2>   -ln m3   ln m6/ln m3   ln m9/ln m3")
for T in sorted(D):
    Z=np.concatenate([d[0] for d in D[T]]); P=np.concatenate([d[1] for d in D[T]]); A=np.concatenate([d[2] for d in D[T]])
    for lo,hi in [(0.95,1.01),(0.85,0.95),(0.6,0.85),(0,0.6)]:
        s=(A>=lo)&(A<hi)
        if s.sum()<20: continue
        z=Z[s].ravel(); p=P[s].ravel(); m3,m6,m9=[np.mean(np.cos(k*p)) for k in (3,6,9)]
        print(f" {T:.4f} | [{lo:.2f},{hi:.2f})  {s.mean():.2f}  | {np.mean(np.arcsin(np.clip(z,-1,1))**2):.4f}   {-np.log(m3):.4f}   {np.log(m6)/np.log(m3):5.2f}        {np.log(m9)/np.log(m3):5.2f}")
