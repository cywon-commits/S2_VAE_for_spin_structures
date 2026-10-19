import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.special import sph_harm_y
# Monte Carlo-free: count invariant real harmonics by projecting with group average on Lebedev-like random points
rng=np.random.default_rng(0)
P=rng.normal(size=(4000,3)); P/=np.linalg.norm(P,axis=1)[:,None]
def Ylm_real(l,pts):
    th=np.arccos(np.clip(pts[:,2],-1,1)); ph=np.arctan2(pts[:,1],pts[:,0])
    out=[]
    for m in range(-l,l+1):
        Y=sph_harm_y(l,abs(m),th,ph)
        if m<0: out.append(np.sqrt(2)*Y.imag)
        elif m==0: out.append(Y.real)
        else: out.append(np.sqrt(2)*Y.real)
    return np.array(out).T
def Rz(a): return Rot.from_euler('z',a).as_matrix()
def Rx(a): return Rot.from_euler('x',a).as_matrix()
def gen_group(gens):
    G=[np.eye(3)]
    changed=True
    while changed:
        changed=False
        for g in list(G):
            for h in gens:
                k=g@h
                if not any(np.allclose(k,x) for x in G):
                    G.append(k); changed=True
    return G
I=-np.eye(3)
groups={
 'D_inf_h (approx D60h)':gen_group([Rz(2*np.pi/60),Rx(np.pi),I]),
 'D3h':gen_group([Rz(2*np.pi/3),Rx(np.pi),I@Rz(np.pi)]),   # sigma_h = -C2z
 'D3d':gen_group([Rz(2*np.pi/3),Rx(np.pi),I]),
 'C3v':gen_group([Rz(2*np.pi/3),I@Rx(np.pi)]),             # sigma_v = -C2x
 'C6v':gen_group([Rz(2*np.pi/6),I@Rx(np.pi)]),
 'C_inf_v (approx C60v)':gen_group([Rz(2*np.pi/60),I@Rx(np.pi)]),
}
for name,G in groups.items():
    counts=[]
    for l in range(0,7):
        Y=Ylm_real(l,P)
        # representation matrices by least squares: Y(g^-1 p) = Y(p) D
        Davg=np.zeros((2*l+1,2*l+1))
        for g in G:
            Yg=Ylm_real(l,P@g)  # f(g^T p)
            D,*_=np.linalg.lstsq(Y,Yg,rcond=None)
            Davg+=D
        Davg/=len(G)
        counts.append(int(round(np.trace(Davg))))
    print(f"{name:24s} |G|={len(G):4d}  n_l(l=0..6) = {counts}")
