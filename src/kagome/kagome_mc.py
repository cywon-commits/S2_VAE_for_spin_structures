"""Classical Heisenberg AF on kagome, H = J sum n_i.n_j (J=1). Sublattice heat bath (exact, linear field) + over-relaxation."""
import numpy as np, sys, time
# neighbors: (sublattice, dx, dy) for each sublattice; positions A=0, B=a1/2, C=a2/2
NB={0:[(1,0,0),(1,-1,0),(2,0,0),(2,0,-1)],
    1:[(0,0,0),(0,1,0),(2,0,0),(2,1,-1)],
    2:[(0,0,0),(0,0,1),(1,0,0),(1,-1,1)]}
def nbsum(n,s):
    return sum(np.roll(n[:,:,t],(-dx,-dy),axis=(0,1)) for t,dx,dy in NB[s])
def heatbath(h,rng):
    k=np.linalg.norm(h,axis=-1); a=h/k[...,None]
    u=rng.uniform(size=k.shape); x=1+np.log(u+(1-u)*np.exp(-2*k))/k
    t=np.where(np.abs(a[...,0:1])<0.9,np.array([1.,0,0]),np.array([0,1.,0]))
    e1=np.cross(a,t); e1/=np.linalg.norm(e1,axis=-1,keepdims=True); e2=np.cross(a,e1)
    ph=rng.uniform(0,2*np.pi,k.shape); s=np.sqrt(np.clip(1-x**2,0,1))
    return x[...,None]*a+s[...,None]*(np.cos(ph)[...,None]*e1+np.sin(ph)[...,None]*e2)
def run(L,T,n_anneal,n_meas,every,n_or,seed):
    rng=np.random.default_rng(seed); n=rng.normal(size=(L,L,3,3)); n/=np.linalg.norm(n,axis=-1,keepdims=True)
    Ts=np.geomspace(0.1,T,n_anneal) if n_anneal>0 else []
    snaps=[]
    for sweep in range(n_anneal+n_meas):
        Tc=Ts[sweep] if sweep<n_anneal else T
        for s in range(3):
            h=-nbsum(n,s)/Tc; n[:,:,s]=heatbath(h,rng)
        for _ in range(n_or):
            for s in range(3):
                f=nbsum(n,s); fh=f/np.linalg.norm(f,axis=-1,keepdims=True)
                n[:,:,s]=2*np.sum(n[:,:,s]*fh,-1,keepdims=True)*fh-n[:,:,s]
        if sweep>=n_anneal and (sweep-n_anneal)%every==0: snaps.append(n.copy())
    return np.array(snaps)
def energy(n):
    return sum(np.sum(n[:,:,s]*nbsum(n,s)) for s in range(3))/2/n[...,0].size
if __name__=="__main__":
    L,T=int(sys.argv[1]),float(sys.argv[2]); t=time.time()
    seed=int(sys.argv[7]) if len(sys.argv)>7 else int(1e5*T)
    S=run(L,T,int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),seed)
    np.save(f"seeds/kag_L{L}_T{T:.4f}_s{seed}.npy",S.astype(np.float32))
    E=np.mean([energy(x) for x in S])
    print(f"L={L} T={T} snaps={len(S)} E/site={E:.5f} (ground -1, harmonic estimate -1+T/2*(1-1/6...)) time={time.time()-t:.0f}s")
