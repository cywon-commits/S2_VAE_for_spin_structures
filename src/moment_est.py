import numpy as np, glob
from numpy.polynomial.legendre import leggauss
from scipy.optimize import fsolve
from scipy.special import eval_legendre as Pl
from samplers import sample_bingham
x,w=leggauss(400)
def M(a,l): p=np.exp(a*Pl(2,x)); return np.sum(w*p*Pl(l,x))/np.sum(w*p)
def est(m2,m4):
    # m_l = exp(-l(l+1) s2/4) * M_l(a)  -> solve for (a, s2)
    f=lambda v:[np.exp(-1.5*v[1])*M(v[0],2)-m2, np.exp(-5*v[1])*M(v[0],4)-m4]
    a,s2=fsolve(f,[4.0,0.03]); return a,s2
rng=np.random.default_rng(3); N=1500000
print("toy validation (true sigma^2 known):")
for a2 in [2.0,4.0,8.0,15.0]:
    for s2 in [0.01,0.04]:
        base=sample_bingham(np.diag([0,0,1.5*a2]),N,rng)
        tx,ty=rng.normal(scale=np.sqrt(s2/2),size=(2,N)); th=np.sqrt(tx**2+ty**2); ph=np.arctan2(ty,tx)
        D=np.stack([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)],1)
        z=np.array([0,0,1.]); ax=np.cross(z,D); s=np.linalg.norm(ax,axis=1); c=D[:,2]
        ax=ax/np.maximum(s,1e-12)[:,None]
        S=base*c[:,None]+np.cross(ax,base)*s[:,None]+ax*(np.sum(ax*base,1)*(1-c))[:,None]
        m2=np.mean(Pl(2,S[:,2])); m4=np.mean(Pl(4,S[:,2])); a,e=est(m2,m4)
        print(f"  a2={a2:5.1f} s2_true={s2:.3f}  -> a_loc={a:6.2f}  s2_est={e:.4f}")
print("\nLL MC data (pooled in global Q frame):")
for f in sorted(glob.glob('ll_L12_T*_o.npy')):
    T=float(f.split('_T')[1][:5]); S=np.load(f).astype(float)
    Q=np.einsum('cxyzi,cxyzj->cij',S,S)/12**3; d=np.linalg.eigh(Q)[1][:,:,-1]
    c=np.abs(np.einsum('cxyzi,ci->cxyz',S,d)).ravel()
    m2=np.mean(Pl(2,c)); m4=np.mean(Pl(4,c)); a,e=est(m2,m4)
    print(f"  T={T:.2f}: a_loc={a:.3f} (mean-field 6S/T={6*m2/T:.3f})  s2_moment={e:.4f}")
