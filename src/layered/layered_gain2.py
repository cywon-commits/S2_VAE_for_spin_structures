import numpy as np, glob, sys, io, contextlib
import os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(HERE,'..')); from fitlib import Fitter
with contextlib.redirect_stdout(io.StringIO()):
    from layered_gain import rot_to_z, cell_triads, CI, NBR, global_triad, rodrigues
def loops_cellframe(X,Ucs,col):
    """loops by colour; chi per site measured in the site's own cell frame (axis = cell cap of axis colour)."""
    out=[]
    for a in range(3):
        mask=col!=a; ua=Ucs[:,a]; uc=Ucs[np.arange(len(X)),col]
        E=uc-np.sum(uc*ua,1,keepdims=True)*ua; E/=np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-9); W=np.cross(ua,E)
        chi=np.arctan2(np.sum(X*W,1),np.sum(X*E,1))
        adj={i:[j for j in NBR[i] if mask[j] and col[j]!=col[i]] for i in np.where(mask)[0]}
        seen=set()
        for i in adj:
            if i in seen: continue
            comp=[];st=[i];ok=True
            while st:
                v=st.pop()
                if v in seen: continue
                seen.add(v); comp.append(v); ok&=len(adj[v])==2; st+=[w for w in adj[v] if w not in seen]
            if ok and len(comp)>=6:
                comp=np.array(comp); out.append((a,comp,np.angle(np.mean(np.exp(1j*chi[comp]))),len(comp)))
    return out
def rot_axis(v,ax,ang):  # per-row axis
    c,s=np.cos(ang),np.sin(ang); return v*c+np.cross(ax,v)*s+ax*np.sum(ax*v,1,keepdims=True)*(1-c)
for T in (0.0005,0.002):
    PB,PC,PC6=[],[],[]; nloops=[]; nhex=[]
    for f in sorted(glob.glob(f'../../data_kag/seeds/kag_L12_T{T:.4f}_s*.npy')):
        for S in np.load(f).astype(float)[::4]:
            X=S.reshape(-1,3); U=global_triad(S); col=np.argmax(X@U.T,1); nu_g=np.linalg.eigh(X.T@X)[1][:,0]
            Uc,nuc=cell_triads(S,U,nu_g); Ucs=Uc[CI]; RB=rot_to_z(Ucs[np.arange(len(X)),col],nuc[CI])
            PB.append(np.einsum('nij,nj->ni',RB,X))
            lp=loops_cellframe(X,Ucs,col); nloops.append(len(lp)); nhex.append(sum(1 for l in lp if l[3]==6))
            Xc=X.copy(); X6=X.copy()
            for a,comp,chib,l in lp:
                Xc[comp]=rot_axis(Xc[comp],Ucs[comp,a],-chib)
                if l==6: X6[comp]=rot_axis(X6[comp],Ucs[comp,a],-chib)
            PC.append(np.einsum('nij,nj->ni',RB,Xc)); PC6.append(np.einsum('nij,nj->ni',RB,X6))
    nb=Fitter(6,order=131,alpha=1e-8).fit(np.concatenate(PB)).nll
    nc=Fitter(6,order=131,alpha=1e-8).fit(np.concatenate(PC)).nll
    n6=Fitter(6,order=131,alpha=1e-8).fit(np.concatenate(PC6)).nll
    L_=np.mean(nloops); H=np.mean(nhex)
    print(f"T={T}: loops/config {L_:.1f} (hexagons {H:.1f})")
    print(f"   B->C all loops : {nb-nc:.4f}/site = {432*(nb-nc):.0f} nats/config  (overfit ~0.5 nat/loop = {0.5*L_:.0f})")
    print(f"   B->C hexagons  : {nb-n6:.4f}/site = {432*(nb-n6):.0f} nats/config  (overfit ~{0.5*H:.0f})")
