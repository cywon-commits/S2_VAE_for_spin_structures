# T1: discarded director spread -> a4 ; test sigma_d^2 = -(35/18) a4/a2^2
import numpy as np
from samplers import sample_bingham
from fitlib import Fitter
rng=np.random.default_rng(11); N=1500000
def small_rot(dirs_sig2,n):
    tx,ty=rng.normal(scale=np.sqrt(dirs_sig2/2),size=(2,n)); th=np.sqrt(tx**2+ty**2); ph=np.arctan2(ty,tx)
    return np.stack([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)],1)
def rot_to(v):  # rotation taking z to v (Rodrigues)
    z=np.array([0,0,1.]); c=v@z; ax=np.cross(z,v); s=np.linalg.norm(ax)
    if s<1e-12: return np.eye(3)
    ax/=s; K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]]); return np.eye(3)+s*K+(1-c)*K@K
print(" a2 | sigma2_true | a2_fit | a4_fit  | a4_pred(-18/35 a2^2 s2) | sigma2_est=-(35/18)a4/a2^2 | odd power")
for a2 in [2.0,4.0]:
  for s2 in [0.0,0.01,0.02,0.04]:
    # base samples from exp(a2 P2(n.z)) = exp(1.5 a2 n_z^2)+const, then rotate each by its own director
    base=sample_bingham(np.diag([0,0,1.5*a2]),N,rng)
    D=small_rot(s2,N) if s2>0 else np.tile([0,0,1.],(N,1))
    # rotate each sample: apply rotation z->d_i  (vectorized Rodrigues)
    z=np.array([0,0,1.]); ax=np.cross(np.tile(z,(N,1)),D); s=np.linalg.norm(ax,axis=1); c=D[:,2]
    ax=np.where(s[:,None]>1e-12,ax/np.maximum(s,1e-12)[:,None],0)
    S=base*c[:,None]+np.cross(ax,base)*s[:,None]+ax*(np.sum(ax*base,1)*(1-c))[:,None]
    f=Fitter(8,alpha=1e-7).fit(S)
    a2f=f.coef(2,0)*np.sqrt(5/(4*np.pi)); a4f=f.coef(4,0)*np.sqrt(9/(4*np.pi))
    P=f.power(); odd=sum(P[l] for l in P if l%2)
    est=-(35/18)*a4f/a2f**2
    print(f" {a2:.0f}  |   {s2:.3f}     | {a2f:6.3f} | {a4f:+.4f} | {-18/35*a2**2*s2:+.4f}                 | {est:+.4f}                    | {odd:.1e}")
