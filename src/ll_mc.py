"""3D simple-cubic Lebwohl-Lasher model H=-J sum P2(n_i.n_j), exact checkerboard heat bath (batched Bingham/ACG)."""
import numpy as np, time, sys
def batched_bingham(A, rng):
    # sample n ∝ exp(n.A.n) for each A[k] (k sites)
    lam,V=np.linalg.eigh(-A)                 # ascending
    lam=lam-lam[:,:1]
    lo=np.full(len(A),1e-12); hi=np.full(len(A),3.0)
    for _ in range(60):
        b=(lo+hi)/2; f=np.sum(1/(b[:,None]+2*lam),1)-1
        lo=np.where(f>0,b,lo); hi=np.where(f>0,hi,b)
    b=(lo+hi)/2; Om=1+2*lam/b[:,None]; logM=-(3-b)/2+1.5*np.log(3/b)
    out=np.empty((len(A),3)); todo=np.arange(len(A))
    while len(todo):
        y=rng.normal(size=(len(todo),3))/np.sqrt(Om[todo]); x=y/np.linalg.norm(y,axis=1)[:,None]
        logr=-np.sum(lam[todo]*x**2,1)+1.5*np.log(np.sum(Om[todo]*x**2,1))-logM[todo]
        acc=np.log(rng.uniform(size=len(todo)))<logr
        out[todo[acc]]=x[acc]; todo=todo[~acc]
    return np.einsum('kij,kj->ki',V,out)
def run(L,T,n_eq,n_meas,every,ordered,seed):
    rng=np.random.default_rng(seed)
    if ordered: n=np.tile([0,0,1.0],(L,L,L,1))+0.05*rng.normal(size=(L,L,L,3))
    else: n=rng.normal(size=(L,L,L,3))
    n/=np.linalg.norm(n,axis=-1,keepdims=True)
    X,Y,Z=np.meshgrid(*[np.arange(L)]*3,indexing='ij'); masks=[((X+Y+Z)%2)==p for p in (0,1)]
    beta=1/T; snaps=[]; Ss=[]
    for sweep in range(n_eq+n_meas):
        for mk in masks:
            Msum=np.zeros((L,L,L,3,3))
            for ax in range(3):
                for s in (1,-1):
                    m=np.roll(n,s,ax); Msum+=m[...,:,None]*m[...,None,:]
            A=1.5*beta*Msum[mk]
            n[mk]=batched_bingham(A,rng)
        if sweep>=n_eq and (sweep-n_eq)%every==0:
            snaps.append(n.astype(np.float32).copy())
    return np.array(snaps)
if __name__=="__main__":
    L=int(sys.argv[1]); T=float(sys.argv[2]); ordered=sys.argv[3]=='o'
    t=time.time(); S=run(L,T,int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),ordered,int(1000*T))
    np.save(f"ll_L{L}_T{T:.3f}_{sys.argv[3]}.npy",S)
    Q=np.einsum('cxyzi,cxyzj->cij',S.astype(float),S.astype(float))/L**3-np.eye(3)/3
    Sval=1.5*np.linalg.eigvalsh(Q)[:,-1]
    print(f"L={L} T={T:.3f} start={'ord' if ordered else 'rnd'}  S={Sval.mean():.3f}±{Sval.std():.3f}  n={len(S)}  time={time.time()-t:.0f}s")
