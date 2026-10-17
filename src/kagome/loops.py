import numpy as np, glob, re, collections
from kagome_mc import NB
def build_nbrs(L):
    nb=np.zeros((L,L,3,4),int)
    idx=lambda x,y,s: ((x%L)*L+(y%L))*3+s
    for x in range(L):
        for y in range(L):
            for s in range(3):
                nb[x,y,s]=[idx(x+dx,y+dy,t) for t,dx,dy in NB[s]]
    return nb.reshape(-1,4)
def global_triad(S):
    X=S.reshape(-1,3); V=np.linalg.eigh(X.T@X)[1]; nu,e1=V[:,0],V[:,2]; e2=np.cross(nu,e1)
    phi=np.arctan2(X@e2,X@e1); th=np.angle(np.mean(np.exp(3j*phi)))/3
    return np.array([np.cos(th+2*np.pi*k/3)*e1+np.sin(th+2*np.pi*k/3)*e2 for k in range(3)])
def loops_in_config(S,NBR):
    X=S.reshape(-1,3); U=global_triad(S); col=np.argmax(X@U.T,1); out=[]
    for a in range(3):
        mask=col!=a; ua=U[a]
        # chi per site: rotation about ua away from ideal direction u_c
        E=U[col]-np.outer(U[col]@ua,ua); E/=np.linalg.norm(E,axis=1,keepdims=True); W=np.cross(ua,E)
        chi=np.arctan2(np.sum(X*W,1),np.sum(X*E,1))
        adj={i:[j for j in NBR[i] if mask[j] and col[j]!=col[i]] for i in np.where(mask)[0]}
        seen=set()
        for i in adj:
            if i in seen: continue
            comp=[];stack=[i];ok=True
            while stack:
                v=stack.pop()
                if v in seen: continue
                seen.add(v); comp.append(v)
                if len(adj[v])!=2: ok=False
                stack+=[w for w in adj[v] if w not in seen]
            if ok and len(comp)>=6:
                c=chi[comp]; m=np.angle(np.mean(np.exp(1j*c)))
                out.append((len(comp),m,np.sqrt(np.mean(np.angle(np.exp(1j*(c-m)))**2))))
    return out
NBR=build_nbrs(12); R=collections.defaultdict(list); valid=collections.Counter(); tot=collections.Counter()
for f in glob.glob('seeds/kag_L12_T*_s*.npy'):
    T=float(re.search(r'T([\d.]+)_s',f).group(1))
    if T>0.0021: continue
    for S in np.load(f).astype(float)[::4]:
        lp=loops_in_config(S,NBR); R[T]+=lp; valid[T]+=sum(l[0] for l in lp); tot[T]+=2*432   # each site in 2 loops
np.save('loops_R.npy',{k:np.array(v) for k,v in R.items()},allow_pickle=True)
for T in sorted(R):
    a=np.array(R[T]); print(f"\nT={T}: loops found {len(a)}, site coverage {valid[T]/tot[T]:.2f}")
    print("  length bin | n loops | <chi_mean^2> (coherent) | <within-loop rms^2> | coherence ratio <|chi_mean|>/<rms>")
    for lo,hi in [(6,7),(8,13),(14,25),(26,61),(61,1000)]:
        s=(a[:,0]>=lo)&(a[:,0]<hi)
        if s.sum()<5: continue
        print(f"  [{lo:3d},{hi:4d}) | {s.sum():6d}  | {np.mean(a[s,1]**2):.4f}                  | {np.mean(a[s,2]**2):.4f}              | {np.mean(np.abs(a[s,1]))/np.mean(a[s,2]):.2f}")
