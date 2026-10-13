import numpy as np
from scipy.integrate import lebedev_rule, quad
from samplers import *
rng=np.random.default_rng(7); N=400000
def moments_quad(logp,funcs,order=131):
    X,W=lebedev_rule(order); X=X.T; lp=logp(X); p=np.exp(lp-lp.max())*W; p/=p.sum()
    return [np.sum(p*f(X)) for f in funcs]
fs=[lambda X:X[:,2], lambda X:X[:,2]**2, lambda X:X[:,0]*X[:,2], lambda X:X[:,0]**2-X[:,1]**2]
def report(name,S,logp):
    ex=moments_quad(logp,fs); mc=[np.mean(f(S)) for f in fs]
    err=max(abs(a-b) for a,b in zip(ex,mc)); se=1/np.sqrt(N)
    print(f"{name:42s} max|MC-exact|={err:.4f}  (MC s.e.~{se:.4f})")
# a. linear field, tilted
h=np.array([1.0,0.5,3.0]); report("linear h=(1,.5,3)",sample_linear(h,N,rng),lambda X:X@h)
h=np.array([0,0,60.0]);  report("linear |h|=60 (low T)",sample_linear(h,N,rng),lambda X:X@h)
# b. uniaxial + axial field
for K,hz in [(-3.0,1.0),(3.0,1.0),(8.0,0.5),(8.0,-12.0)]:
    a=hz/(2*K); x=sample_x_quadratic(K,a,N,rng); S=assemble(x,np.array([0,0,1.0]),rng)
    report(f"uniaxial K={K}, h_z={hz}",S,lambda X:K*X[:,2]**2+hz*X[:,2])
# c. Bingham (P2 conditional, biaxial)
A=np.diag([0.5,-1.0,4.0]); Rm=np.linalg.qr(np.random.default_rng(1).normal(size=(3,3)))[0]; A=Rm@A@Rm.T
S=sample_bingham(A,N,rng); report("Bingham biaxial (rotated)",S,lambda X:np.einsum('ni,ij,nj->n',X,A,X))
A2=np.diag([0,0,15.0]); S=sample_bingham(A2,N,rng); report("Bingham sharp two caps a=15",S,lambda X:15*X[:,2]**2)
# acceptance rates
import time
for K,hz in [(3.0,1.0),(30.0,0.0)]:
    a=hz/(2*K); t=time.time(); x=sample_x_quadratic(K,a,200000,rng); 
