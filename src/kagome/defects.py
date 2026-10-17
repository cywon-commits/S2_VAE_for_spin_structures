"""Classify triad-frame defects on kagome hexagon loops via lifting to the binary dihedral group 2D3."""
import numpy as np, glob, re, collections
from scipy.spatial.transform import Rotation as Rot
def tri_spins(S):
    A,B,C=S[:,:,0],S[:,:,1],S[:,:,2]
    up=np.stack([A,B,C],2)                                            # U(R)={A(R),B(R),C(R)}
    dn=np.stack([A,np.roll(B,1,axis=0),np.roll(C,1,axis=1)],2)        # D(R)={A(R),B(R-a1),C(R-a2)}
    return up,dn
def candidates(T):   # T: (...,3 spins,3) -> quaternions (...,12,4)
    s=T; kap=np.cross(s[...,0,:],s[...,1,:])+np.cross(s[...,1,:],s[...,2,:])+np.cross(s[...,2,:],s[...,0,:])
    kap/=np.linalg.norm(kap,axis=-1,keepdims=True); Q=[]
    for k in range(3):
        for sg in (1,-1):
            e3=sg*kap; e1=s[...,k,:]-np.sum(s[...,k,:]*e3,-1,keepdims=True)*e3
            e1/=np.linalg.norm(e1,axis=-1,keepdims=True); e2=np.cross(e3,e1)
            M=np.stack([e1,e2,e3],-1); q=Rot.from_matrix(M.reshape(-1,3,3)).as_quat().reshape(M.shape[:-2]+(4,))
            Q.append(q); Q.append(-q)
    return np.stack(Q,-2)       # order: (k,sg,±) -> index 4k+2*(sg==-1)+(minus)
def classify(S):
    up,dn=tri_spins(S); CU,CD=candidates(up),candidates(dn); L=S.shape[0]
    r=lambda X,dx,dy: np.roll(X,(-dx,-dy),axis=(0,1))
    loop=[r(CU,0,0),r(CD,1,0),r(CU,1,0),r(CD,1,1),r(CU,0,1),r(CD,0,1),r(CU,0,0)]
    q=loop[0][:,:,0,:]                                    # start: k=0, sg=+, +
    for C in loop[1:]:
        d=np.einsum('xyci,xyi->xyc',C,q); idx=np.argmax(d,-1)
        q=np.take_along_axis(C,idx[...,None,None],2)[:,:,0,:]
    k=idx//4; sg=(idx//2)%2; mn=idx%2
    cls=np.full(idx.shape,'?',dtype=object)
    cls[(k==0)&(sg==0)&(mn==0)]='none'; cls[(k==0)&(sg==0)&(mn==1)]='Z2'
    cls[(k!=0)&(sg==0)]='C3'; cls[sg==1]='C2'
    return cls
D=collections.defaultdict(collections.Counter); N=collections.Counter()
for f in glob.glob('seeds/kag_L12_T*_s*.npy'):
    T=float(re.search(r'T([\d.]+)_s',f).group(1))
    for S in np.load(f).astype(float)[::4]:
        c=classify(S); D[T].update(c.ravel().tolist()); N[T]+=c.size
print(" T      | defect density per hexagon:  Z2       C3 (color)   C2 (plane flip)   total")
for T in sorted(D):
    d=D[T]; n=N[T]
    print(f" {T:.4f} |                            {d['Z2']/n:.4f}   {d['C3']/n:.4f}       {d['C2']/n:.4f}            {(n-d['none'])/n:.4f}")
