import numpy as np, glob, re, collections
def frame(S):
    X=S.reshape(-1,3).astype(float); w,V=np.linalg.eigh(X.T@X/len(X))
    nu,e1=V[:,0],V[:,2]; e2=np.cross(nu,e1)
    z=S@nu; phi=np.arctan2(S@e2,S@e1)
    psi=np.mean(np.exp(3j*phi)); return z,phi-np.angle(psi)/3
def classes(S):
    z,p=frame(S.astype(float)); col=np.round(3*p/(2*np.pi)).astype(int)%3   # nearest triad direction
    L=S.shape[0]; r=lambda a,dx,dy: np.roll(a,(-dx,-dy),axis=(0,1))
    # hexagon at R: C(R),B(R),A(R+a1),C(R+a1),B(R+a2),A(R+a2)
    ring=[r(col[:,:,2],0,0),r(col[:,:,1],0,0),r(col[:,:,0],1,0),r(col[:,:,2],1,0),r(col[:,:,1],0,1),r(col[:,:,0],0,1)]
    ring=np.stack(ring,-1)
    flip=(ring[...,0]==ring[...,2])&(ring[...,2]==ring[...,4])&(ring[...,1]==ring[...,3])&(ring[...,3]==ring[...,5])&(ring[...,0]!=ring[...,1])
    mark=np.zeros(col.shape,bool)
    for (s,dx,dy) in [(2,0,0),(1,0,0),(0,1,0),(2,1,0),(1,0,1),(0,0,1)]:
        # site (s) at position R+(dx,dy) belongs to hexagon R: roll flip back
        mark[:,:,s]|=np.roll(flip,(dx,dy),axis=(0,1))
    return z,p,mark,flip.mean()
D=collections.defaultdict(list)
for f in glob.glob('seeds/kag_L12_T*_s*.npy'):
    T=float(re.search(r'T([\d.]+)_s',f).group(1))
    for S in np.load(f)[::4]: D[T].append(classes(S))
print(" T      | flippable hex frac | site frac in flippable | class:     <eps^2>   -ln m3   ln m6/ln m3")
for T in sorted(D):
    Z=np.concatenate([d[0].ravel() for d in D[T]]); P=np.concatenate([d[1].ravel() for d in D[T]])
    M=np.concatenate([d[2].ravel() for d in D[T]]); hf=np.mean([d[3] for d in D[T]])
    line=f" {T:.4f} | {hf:.3f}              | {M.mean():.3f}                 |"
    for name,sel in [("flippable",M),("locked",~M)]:
        e2=np.mean(np.arcsin(np.clip(Z[sel],-1,1))**2); m3=np.mean(np.cos(3*P[sel])); m6=np.mean(np.cos(6*P[sel]))
        line+=f" {name}: {e2:.4f} {-np.log(m3):.4f} {np.log(m6)/np.log(m3):.2f} |"
    print(line)
