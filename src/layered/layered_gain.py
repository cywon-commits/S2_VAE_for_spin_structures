"""Layered information decomposition given the colouring: A (global caps) -> B (cell-frame caps) -> C (B + loop rotation)."""
import numpy as np, glob, re, sys
import os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(HERE,'..')); from fitlib import Fitter
from loops import build_nbrs, global_triad
NBR=build_nbrs(12); L=12; b=3
def rot_to_z(u,nu):   # rows: x'=nu x u, y'=nu, z'=u  -> local coords
    x=np.cross(nu,u); x/=np.linalg.norm(x,axis=-1,keepdims=True); y=np.cross(u,x); return np.stack([x,y,u],-2)
def cell_triads(S,U,nu_g):
    c=S.reshape(L//b,b,L//b,b,3,3).transpose(0,2,1,3,4,5).reshape(-1,3*b*b,3)
    Q=np.einsum('cni,cnj->cij',c,c); V=np.linalg.eigh(Q)[1]; nu=V[:,:,0]; nu*=np.sign(nu@nu_g)[:,None]
    e1=V[:,:,2]; e2=np.cross(nu,e1); phi=np.arctan2(np.einsum('cni,ci->cn',c,e2),np.einsum('cni,ci->cn',c,e1))
    th=np.angle(np.mean(np.exp(3j*phi),1))/3; Uc=np.zeros((len(c),3,3))
    caps=np.stack([np.cos(th[:,None]+2*np.pi*k/3)*e1+np.sin(th[:,None]+2*np.pi*k/3)*e2 for k in range(3)],1)  # cell x 3 x 3
    for k in range(3):   # match each global colour to nearest cell cap
        j=np.argmax(np.einsum('cki,i->ck',caps,U[k]),1); Uc[:,k]=caps[np.arange(len(c)),j]
    return Uc,nu
def site_cell_index():
    x,y=np.meshgrid(np.arange(L),np.arange(L),indexing='ij'); ci=((x//b)*(L//b)+(y//b))
    return np.repeat(ci[:,:,None],3,2).ravel()
CI=site_cell_index()
def loop_rotations(X,U,col):
    """return per-site list of (axis, chi_bar) for its loops (as in loops.py)"""
    corr=[]
    for a in range(3):
        mask=col!=a; ua=U[a]
        E=U[col]-np.outer(U[col]@ua,ua); nrm=np.linalg.norm(E,axis=1,keepdims=True); E=E/np.maximum(nrm,1e-9); W=np.cross(ua,E)
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
            if ok and len(comp)>=6: corr.append((a,np.array(comp),np.angle(np.mean(np.exp(1j*chi[comp])))))
    return corr
def rodrigues(v,axis,ang):
    c,s=np.cos(ang),np.sin(ang); return v*c+np.cross(axis,v)*s+np.outer(v@axis,axis)*(1-c) if v.ndim==2 else None
def main():
  for T in (0.0005,0.002):
      PA,PB,PC=[],[],[]
      for f in sorted(glob.glob(f'../../data_kag/seeds/kag_L12_T{T:.4f}_s*.npy')):
          for S in np.load(f).astype(float)[::4]:
              X=S.reshape(-1,3); U=global_triad(S); col=np.argmax(X@U.T,1)
              Xg=X.reshape(-1,3); xx=X.T@X; nu_g=np.linalg.eigh(xx)[1][:,0]
              # A: global caps
              RA=rot_to_z(U[col],np.tile(nu_g,(len(X),1))); PA.append(np.einsum('nij,nj->ni',RA,X))
              # B: cell caps
              Uc,nuc=cell_triads(S,U,nu_g); uc=Uc[CI,col]; RB=rot_to_z(uc,nuc[CI]); PB.append(np.einsum('nij,nj->ni',RB,X))
              # C: undo loop rotations (global axes), then cell caps
              Xc=X.copy()
              for a,comp,chib in loop_rotations(X,U,col):
                  Xc[comp]=rodrigues(Xc[comp],U[a],-chib)
              PC.append(np.einsum('nij,nj->ni',RB,Xc))
      res={}
      for name,P in (('A',PA),('B',PB),('C',PC)):
          res[name]=Fitter(6,order=131,alpha=1e-8).fit(np.concatenate(P)).nll
      nconf=len(PA)
      print(f"T={T}: configs={nconf}  NLL/site A={res['A']:.4f} B={res['B']:.4f} C={res['C']:.4f}")
      print(f"   gain A->B (local frame, R=4 cells) = {res['A']-res['B']:.4f}/site = {432*(res['A']-res['B']):.0f} nats/config")
      print(f"   gain B->C (loop rotations)         = {res['B']-res['C']:.4f}/site = {432*(res['B']-res['C']):.0f} nats/config")

if __name__=='__main__': main()
