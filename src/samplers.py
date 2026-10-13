"""Exact single-site samplers on S^2 (heat bath building blocks)."""
import numpy as np
from scipy.stats import truncnorm

def _frame(axis):
    a = axis/np.linalg.norm(axis)
    t = np.array([1.0,0,0]) if abs(a[0])<0.9 else np.array([0,1.0,0])
    e1 = np.cross(a,t); e1/=np.linalg.norm(e1); e2=np.cross(a,e1)
    return a,e1,e2

def assemble(x, axis, rng):
    """cos(theta)=x about axis, azimuth uniform."""
    a,e1,e2=_frame(axis); phi=rng.uniform(0,2*np.pi,len(x)); s=np.sqrt(np.clip(1-x**2,0,1))
    return x[:,None]*a + s[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)

def sample_linear(h, n, rng):
    """p ∝ exp(h·n) (beta absorbed). Inverse CDF, stable form."""
    k=np.linalg.norm(h); u=rng.uniform(size=n)
    x = 1+np.log(u+(1-u)*np.exp(-2*k))/k
    return assemble(x,h,rng)

def sample_x_quadratic(K, a, n, rng):
    """p(x) ∝ exp(K (x+a)^2), x in [-1,1]. K<0: truncated Gaussian. K>0: exact rejection
    with piecewise chord (upper) envelopes on [-1,x0] and [x0,1], x0=clip(-a)."""
    if K<0:
        sd=np.sqrt(1/(2*(-K))); mu=-a
        return truncnorm.rvs((-1-mu)/sd,(1-mu)/sd,loc=mu,scale=sd,size=n,random_state=rng)
    logf=lambda x: K*(x+a)**2
    x0=np.clip(-a,-1,1)
    segs=[(lo,hi) for lo,hi in [(-1.0,x0),(x0,1.0)] if hi-lo>1e-12]
    # convex log f -> chord is an upper bound; envelope exp(linear) on each segment
    env=[]
    for lo,hi in segs:
        s=(logf(hi)-logf(lo))/(hi-lo); c=logf(lo)-s*lo
        mass=(np.exp(s*hi+c)-np.exp(s*lo+c))/s if abs(s)>1e-12 else np.exp(c)*(hi-lo)
        env.append((lo,hi,s,c,mass))
    masses=np.array([e[4] for e in env]); probs=masses/masses.sum()
    out=np.empty(0)
    while len(out)<n:
        m=2*(n-len(out))+10
        idx=rng.choice(len(env),size=m,p=probs); u=rng.uniform(size=m); v=rng.uniform(size=m)
        xs=np.empty(m); acc=np.empty(m,bool)
        for j,(lo,hi,s,c,_) in enumerate(env):
            sel=idx==j
            if abs(s)>1e-12:
                xs[sel]=np.log(np.exp(s*lo)+u[sel]*(np.exp(s*hi)-np.exp(s*lo)))/s
            else: xs[sel]=lo+u[sel]*(hi-lo)
            acc[sel]=np.log(v[sel]) < logf(xs[sel])-(s*xs[sel]+c)
        out=np.concatenate([out,xs[acc]])
    return out[:n]

def sample_bingham(A, n, rng):
    """p ∝ exp(n·A·n), exact (Kent, Ganeiber & Mardia 2018, ACG envelope, q=3)."""
    lam,V=np.linalg.eigh(-A); lam=lam-lam.min()        # exp(-x^T L x), L>=0
    from scipy.optimize import brentq
    b=1.0 if np.allclose(lam,0) else brentq(lambda b: np.sum(1/(b+2*lam))-1,1e-10,3.0)
    Om=1+2*lam/b; M=np.exp(-(3-b)/2)*(3/b)**1.5
    out=[]; got=0
    while got<n:
        m=2*(n-got)+10
        y=rng.normal(size=(m,3))/np.sqrt(Om); x=y/np.linalg.norm(y,axis=1)[:,None]
        q=np.sum(lam*x**2,1); r=np.sum(Om*x**2,1)
        acc=rng.uniform(size=m) < np.exp(-q)*r**1.5/M
        out.append(x[acc]); got+=acc.sum()
    return (np.concatenate(out)[:n])@V.T
