import numpy as np, glob, re, collections
rng=np.random.default_rng(0)
def cell_stats(S,b):
    L=S.shape[0]; c=S.reshape(L//b,b,L//b,b,3,3).transpose(0,2,1,3,4,5).reshape(-1,3*b*b,3)  # cells x sites x 3
    Q=np.einsum('cni,cnj->cij',c,c); w,V=np.linalg.eigh(Q); nu=V[:,:,0]; e1=V[:,:,2]; e2=np.cross(nu,e1)
    z=np.einsum('cni,ci->cn',c,nu); phi=np.arctan2(np.einsum('cni,ci->cn',c,e2),np.einsum('cni,ci->cn',c,e1))
    psi=np.mean(np.exp(3j*phi),1); p=phi-np.angle(psi)[:,None]/3
    return z.ravel(),p.ravel(),nu
def glob_nu(S):
    X=S.reshape(-1,3); return np.linalg.eigh(X.T@X)[1][:,0]
D=collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob('seeds/kag_L12_T*_s*.npy'):
    T=float(re.search(r'T([\d.]+)_s',f).group(1))
    if T>0.0041: continue
    for S in np.load(f).astype(float)[::4]:
        g=glob_nu(S); D[T]['glob'].append(cell_stats(S,12)[:2])
        Ssh=S.reshape(-1,3)[rng.permutation(432)].reshape(S.shape)
        for b in (2,3,4,6):
            z,p,nu=cell_stats(S,b); D[T][b].append((z,p,np.mean(np.arccos(np.clip(np.abs(nu@g),0,1))**2)))
            zs,ps,_=cell_stats(Ssh,b); D[T][('sh',b)].append((zs,ps,0))
def mom(lst):
    Z=np.concatenate([x[0] for x in lst]); P=np.concatenate([x[1] for x in lst])
    e2=np.mean(np.arcsin(np.clip(Z,-1,1))**2); m3=np.mean(np.cos(3*P)); m6=np.mean(np.cos(6*P)); m9=np.mean(np.cos(9*P))
    return e2,-np.log(m3),np.log(m6)/np.log(m3),np.log(m9)/np.log(m3)
print("frame (sites/cell) |   T   | <eps^2>  -ln m3  ln m6/ln m3 (Gauss 4)  ln m9/ln m3 (Gauss 9) | cell plane tilt^2 | shuffled <eps^2>, -ln m3")
for T in sorted(D):
    e=mom(D[T]['glob']); print(f" global (432)       | {T:.4f} | {e[0]:.4f}  {e[1]:.4f}   {e[2]:.2f}                  {e[3]:.2f}")
    for b in (6,4,3,2):
        e=mom(D[T][b]); tilt=np.mean([x[2] for x in D[T][b]]); s=mom(D[T][('sh',b)])
        print(f" cell b={b} ({3*b*b:3d})     | {T:.4f} | {e[0]:.4f}  {e[1]:.4f}   {e[2]:.2f}                  {e[3]:.2f}     | {tilt:.4f}           | {s[0]:.4f}, {s[1]:.4f}")
