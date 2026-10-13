import numpy as np
from scipy.integrate import lebedev_rule
from numpy.polynomial.legendre import leggauss
def lz_leb(f,order):
    X,W=lebedev_rule(order); v=f(X.T); m=v.max(); return m+np.log(np.sum(W*np.exp(v-m)))
def lz_product(f,axis,nx=64,nphi=64):
    # Gauss-Legendre in x=cos(theta) about axis (mode-aligned), uniform phi
    a=axis/np.linalg.norm(axis); t=np.array([1.,0,0]) if abs(a[0])<.9 else np.array([0,1.,0])
    e1=np.cross(a,t); e1/=np.linalg.norm(e1); e2=np.cross(a,e1)
    x,w=leggauss(nx); ph=np.arange(nphi)*2*np.pi/nphi; s=np.sqrt(1-x**2)
    X=(x[:,None,None]*a+ s[:,None,None]*(np.cos(ph)[None,:,None]*e1+np.sin(ph)[None,:,None]*e2)).reshape(-1,3)
    Wt=(w[:,None]*np.ones(nphi)*2*np.pi/nphi).ravel(); v=f(X); m=v.max(); return m+np.log(np.sum(Wt*np.exp(v-m)))
exact_vmf=lambda k: np.log(4*np.pi*np.sinh(k)/k) if k<300 else k+np.log(2*np.pi/k)
axis=np.array([0.3,-0.5,0.81]); axis/=np.linalg.norm(axis)
print("vMF  |  kappa | Leb 17 | Leb 35 | Leb 59 | Leb 131 | product(64x64, mode-aligned)")
for k in [1,5,20,50,100,300,1000]:
    f=lambda X: k*(X@axis)
    errs=[abs(lz_leb(f,o)-exact_vmf(k)) for o in [17,35,59,131]]
    ep=abs(lz_product(f,axis)-exact_vmf(k))
    print(f"      {k:6d} | "+" | ".join(f"{e:.1e}" for e in errs)+f" | {ep:.1e}")
# two sharp caps (Bingham a n_z^2): reference = 1D integral 2pi int exp(a x^2)dx
from scipy.integrate import quad
print("\nTwo caps exp(a (n.e)^2) | a | Leb 35 | Leb 131 | product(64x64)")
for a in [5,20,100,500]:
    ref=np.log(2*np.pi*quad(lambda x: np.exp(a*x**2-a),-1,1,points=[-1+1e-3,1-1e-3],limit=200)[0])+a
    f=lambda X: a*(X@axis)**2
    print(f"   {a:5d} | {abs(lz_leb(f,35)-ref):.1e} | {abs(lz_leb(f,131)-ref):.1e} | {abs(lz_product(f,axis)-ref):.1e}")
# grid-convergence criterion: degree needed so that error<1e-3, vs kappa
print("\nminimal Lebedev degree for |err|<1e-3 (vMF):")
orders=[3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,35,41,47,53,59,65,71,77,83,89,95,101,107,113,119,125,131]
for k in [1,5,10,20,50,100,200]:
    f=lambda X: k*(X@axis)
    ok=[o for o in orders if abs(lz_leb(f,o)-exact_vmf(k))<1e-3]
    print(f"   kappa={k:4d}: degree {ok[0] if ok else '>131'}")
