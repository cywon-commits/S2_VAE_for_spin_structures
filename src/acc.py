import numpy as np
from samplers import *
from scipy.optimize import brentq
rng=np.random.default_rng(3)
# chord-envelope acceptance for easy-axis K>0
def acc_quad(K,a):
    logf=lambda x: K*(x+a)**2; x0=np.clip(-a,-1,1)
    from scipy.integrate import quad
    Z=quad(lambda x: np.exp(logf(x)-K*4),-1,1,points=[x0])[0]
    env=0
    for lo,hi in [(-1,x0),(x0,1)]:
        if hi-lo<1e-12: continue
        s=(logf(hi)-logf(lo))/(hi-lo); c=logf(lo)-s*lo-K*4
        env+= (np.exp(s*hi+c)-np.exp(s*lo+c))/s if abs(s)>1e-12 else np.exp(c)*(hi-lo)
    return Z/env
for K,hz in [(1,0),(3,0),(10,0),(30,0),(100,0),(10,5),(10,-30)]:
    print(f"chord K={K:5.0f} h_z={hz:5.0f}: acceptance={acc_quad(K,hz/(2*K)):.3f}")
def acc_bing(lams):
    lam=np.array(lams,float); lam=lam-lam.min()
    b=1.0 if np.allclose(lam,0) else brentq(lambda b: np.sum(1/(b+2*lam))-1,1e-10,3.0)
    Om=1+2*lam/b; M=np.exp(-(3-b)/2)*(3/b)**1.5
    y=rng.normal(size=(400000,3))/np.sqrt(Om); x=y/np.linalg.norm(y,axis=1)[:,None]
    return np.mean(np.exp(-np.sum(lam*x**2,1))*np.sum(Om*x**2,1)**1.5/M)
for L in [[0,0,1],[0,0,10],[0,0,100],[0,5,50],[0,50,50]]:
    print(f"Bingham ACG eigen(-A)={L}: acceptance={acc_bing(L):.3f}")
