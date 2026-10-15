import numpy as np
from fitlib import Fitter
rng=np.random.default_rng(1)
def frames_global(S):
    Q=np.einsum('cxyzi,cxyzj->cij',S,S); return np.linalg.eigh(Q)[1][:,:,[1,0,2]]
def rotate_cells(S,b):
    c,L=S.shape[0],S.shape[1]; Sb=S.reshape(c,L//b,b,L//b,b,L//b,b,3)
    Qb=np.einsum('cxaybzdi,cxaybzdj->cxyzij',Sb,Sb); V=np.linalg.eigh(Qb)[1][...,[1,0,2]]
    return np.einsum('cxaybzdi,cxyzij->cxaybzdj',Sb,V).reshape(-1,3)
def nll(P): return Fitter(4,order=59,alpha=1e-8).fit(P).nll
S=np.load('ll_L12_T0.950_o.npy').astype(float)[:100]; N=12**3
Sg=np.einsum('cxyzi,cij->cxyzj',S,frames_global(S)).reshape(-1,3); base=nll(Sg)
Ssh=S.reshape(len(S),-1,3).copy()
for k in range(len(S)): Ssh[k]=Ssh[k][rng.permutation(N)]
Ssh=Ssh.reshape(S.shape)
Sgs=np.einsum('cxyzi,cij->cxyzj',Ssh,frames_global(Ssh)).reshape(-1,3); base_s=nll(Sgs)
print("T=0.95: NLL gain from exact cell frames (nats/site)")
print(" R  n   raw gain  overfit(shuffled)  net gain | corrected pred | VAE indep(v2) | VAE hier(v3) | hier early(750 steps)")
pred={2:0.043,3:0.094,4:0.144}; v2={2:0.038,3:0.087,4:0.111}; v3={2:0.002,3:0.008,4:0.012}
for R,b in [(2,6),(3,4),(4,3)]:
    g=base-nll(rotate_cells(S,b)); o=base_s-nll(rotate_cells(Ssh,b))
    print(f" {R}  {b**3:3d}  {g:.4f}     {o:.4f}            {g-o:.4f}  |    {pred[R]:.3f}      |    {v2[R]:.3f}      |   {v3[R]:.3f}     |  {'0.131' if R==4 else ''}")
