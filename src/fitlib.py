import numpy as np
from scipy.special import sph_harm_y
from scipy.integrate import lebedev_rule
from scipy.optimize import minimize
def real_sh(L,X):
    th=np.arccos(np.clip(X[:,2],-1,1)); ph=np.arctan2(X[:,1],X[:,0]); cols=[]; idx=[]
    for l in range(1,L+1):
        for m in range(-l,l+1):
            Y=sph_harm_y(l,abs(m),th,ph)
            cols.append(np.sqrt(2)*(-1)**m*Y.imag if m<0 else (Y.real if m==0 else np.sqrt(2)*(-1)**m*Y.real)); idx.append((l,m))
    return np.array(cols).T, idx
class Fitter:
    """Unrestricted exponential family p ∝ exp(sum lam_lm Y_lm), l=1..L, Laplace-Beltrami penalty."""
    def __init__(s,L,order=131,alpha=1e-6):
        s.L=L; X,W=lebedev_rule(order); s.Yq,s.idx=real_sh(L,X.T); s.W=W; s.alpha=alpha
        s.pen=np.array([l*(l+1) for l,m in s.idx],float)
    def fit(s,S,chunk=200000):
        mY=np.zeros(len(s.idx))
        for i in range(0,len(S),chunk): mY+=real_sh(s.L,S[i:i+chunk])[0].sum(0)
        mY/=len(S); s.mY=mY
        def obj(lam):
            v=s.Yq@lam; m=v.max(); e=s.W*np.exp(v-m); Z=e.sum(); p=e/Z
            f=-mY@lam+m+np.log(Z)+s.alpha*np.sum(s.pen*lam**2)
            g=-mY+s.Yq.T@p+2*s.alpha*s.pen*lam
            return f,g
        r=minimize(obj,np.zeros(len(s.idx)),jac=True,method='L-BFGS-B',options={'maxiter':5000,'gtol':1e-9})
        s.lam=r.x; s.nll=r.fun-s.alpha*np.sum(s.pen*r.x**2); return s
    def power(s):
        P={}
        for (l,m),v in zip(s.idx,s.lam): P[l]=P.get(l,0)+v**2
        return P
    def coef(s,l,m): return s.lam[s.idx.index((l,m))]
    def E(s,X): return -(real_sh(s.L,X)[0]@s.lam)
