import numpy as np, glob, re
from defects import classify
from scipy.spatial.transform import Rotation as Rot
rng=np.random.default_rng(0); L=12
u=np.array([[1,0,0],[np.cos(2*np.pi/3),np.sin(2*np.pi/3),0],[np.cos(4*np.pi/3),np.sin(4*np.pi/3),0]])
def q0(): S=np.zeros((L,L,3,3)); S[:]=u; return S                       # q=0 state
def sqrt3():                                                             # sqrt3 x sqrt3 state
    S=np.zeros((L,L,3,3)); 
    for x in range(L):
        for y in range(L):
            m=(2*x+y)%3; S[x,y,0]=u[m]; S[x,y,1]=u[(m+1)%3]; S[x,y,2]=u[(m+2)%3]
    return S
def noisy(S,s): S=S+s*rng.normal(size=S.shape); return S/np.linalg.norm(S,axis=-1,keepdims=True)
def smooth_twist(S,amp):
    X,Y=np.meshgrid(np.arange(L),np.arange(L),indexing='ij')
    ang=amp*np.stack([np.sin(2*np.pi*X/L),np.cos(2*np.pi*Y/L),np.sin(2*np.pi*(X+Y)/L)],-1)
    R=Rot.from_rotvec(ang.reshape(-1,3)).as_matrix().reshape(L,L,3,3)
    return np.einsum('xyij,xysj->xysi',R,S)
for name,S in [("q=0",q0()),("sqrt3",sqrt3())]:
    for s in (0.05,0.15,0.3):
        c=classify(noisy(S,s)); print(f"{name:6s} noise {s:.2f}: nontrivial frac {np.mean(c!='none'):.4f}")
    c=classify(noisy(smooth_twist(S,0.8),0.05)); print(f"{name:6s} smooth twist amp 0.8 + noise: {np.mean(c!='none'):.4f}")
# correlation with local octupolar order (27-site cells = 3x3 unit cells -> hexagons in the same block)
from defect_check import cells
print("\nMC: defect density in hexagons vs local octupolar order of their 3x3 block")
for f in sorted(glob.glob('seeds/kag_L12_T0.0020_s*.npy'))[:3]:
    for S in np.load(f).astype(float)[::20]:
        c=classify(S); _,_,A=cells(S,3); dens=(c!='none').reshape(4,3,4,3).mean(axis=(1,3)).ravel()
        if 'acc' not in dir(): acc=[]
        acc+=list(zip(A,dens))
a=np.array(acc)
for lo,hi in [(0.95,1.01),(0.85,0.95),(0.6,0.85),(0,0.6)]:
    s=(a[:,0]>=lo)&(a[:,0]<hi); print(f"  |psi_cell| in [{lo:.2f},{hi:.2f}): n={s.sum():4d}  defect density {a[s,1].mean():.3f}")
