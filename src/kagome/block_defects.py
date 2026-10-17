import numpy as np, glob, re, collections
from scipy.spatial.transform import Rotation as Rot
def block_frames(S,b):
    L=S.shape[0]; c=S.reshape(L//b,b,L//b,b,3,3).transpose(0,2,1,3,4,5).reshape(L//b,L//b,-1,3)
    Q=np.einsum('xyni,xynj->xyij',c,c); V=np.linalg.eigh(Q)[1]; nu=V[...,0]; e1=V[...,2]; e2=np.cross(nu,e1)
    phi=np.arctan2(np.einsum('xyni,xyi->xyn',c,e2),np.einsum('xyni,xyi->xyn',c,e1))
    psi=np.mean(np.exp(3j*phi),-1); th=np.angle(psi)/3; Qs=[]
    for k in range(3):
        a=th+2*np.pi*k/3; u=np.cos(a)[...,None]*e1+np.sin(a)[...,None]*e2
        for sg in (1,-1):
            e3=sg*nu; M=np.stack([u,np.cross(e3,u),e3],-1)
            q=Rot.from_matrix(M.reshape(-1,3,3)).as_quat().reshape(M.shape[:-2]+(4,)); Qs+=[q,-q]
    return np.stack(Qs,-2),np.abs(psi)
def classify_blocks(S,b):
    C,A=block_frames(S,b); r=lambda X,dx,dy: np.roll(X,(-dx,-dy),axis=(0,1))
    loop=[C,r(C,1,0),r(C,1,1),r(C,0,1),C]; q=C[:,:,0,:]
    for X in loop[1:]:
        idx=np.argmax(np.einsum('xyci,xyi->xyc',X,q),-1); q=np.take_along_axis(X,idx[...,None,None],2)[:,:,0,:]
    k=idx//4; sg=(idx//2)%2; mn=idx%2
    out=np.where((k==0)&(sg==0)&(mn==0),'none',np.where((k==0)&(sg==0),'Z2',np.where(sg==0,'C3','C2')))
    return out
D=collections.defaultdict(collections.Counter); N=collections.Counter()
for f in glob.glob('seeds/kag_L12_T*_s*.npy'):
    T=float(re.search(r'T([\d.]+)_s',f).group(1))
    for S in np.load(f).astype(float)[::4]:
        for b in (2,3):
            c=classify_blocks(S,b); D[(T,b)].update(c.ravel().tolist()); N[(T,b)]+=c.size
print(" T      | block | loops/config | per-loop density: Z2     C3      C2     | defects per config (Z2+C3+C2)")
for (T,b) in sorted(D):
    d=D[(T,b)]; n=N[(T,b)]; per=(12//b)**2
    print(f" {T:.4f} |  {b}x{b}  |    {per:3d}       |                  {d['Z2']/n:.4f} {d['C3']/n:.4f} {d['C2']/n:.4f} | {per*(n-d['none'])/n:.2f}")
