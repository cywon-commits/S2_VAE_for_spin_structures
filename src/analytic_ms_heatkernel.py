import numpy as np
from scipy.special import sph_harm_y, eval_legendre
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
np.set_printoptions(precision=4, suppress=True)
x,w = leggauss(400)
# ---------- 1. Maier-Saupe ----------
def Zf(a): return np.sum(w*np.exp(a*eval_legendre(2,x)))/2
def Sav(a): return np.sum(w*eval_legendre(2,x)*np.exp(a*eval_legendre(2,x)))/2/Zf(a)
# a = S/t with t = T/(zJ);  f/(zJ) = S^2/2 - t ln Z(S/t)
def Ssol(t):
    g=lambda S: Sav(S/t)-S
    Ss=np.linspace(1e-3,0.999,2000); vals=[g(s) for s in Ss]
    roots=[brentq(g,Ss[i],Ss[i+1]) for i in range(len(Ss)-1) if vals[i]*vals[i+1]<0]
    return max(roots) if roots else 0.0
def fdiff(t):
    S=Ssol(t)
    if S<1e-6: return 1.0
    return S**2/2 - t*np.log(Zf(S/t))
tc=brentq(fdiff,0.20,0.2215)
Sc=Ssol(tc)
print("MS: T_NI/(zJ)=%.4f  S_c=%.4f  a2_c=%.4f"%(tc,Sc,Sc/tc))
for t in [0.10,0.15,0.20,tc-1e-4]:
    S=Ssol(t); print("  t=%.4f S=%.4f a2=%.3f"%(t,S,S/t))
# ---------- 2. heat-kernel check: director spread generates a4 ----------
# p_sigma(n) = < exp(a2 P2(n.d)) >_d, d small isotropic spread sigma^2 = <theta^2>
def a_coeffs(a2,sig2,lmax=6,Nd=200000):
    rng=np.random.default_rng(1)
    # sample d: small rotation of z by angle theta with 2D gaussian (theta_x,theta_y)
    tx,ty=rng.normal(scale=np.sqrt(sig2/2),size=(2,Nd))
    th=np.sqrt(tx**2+ty**2); ph=np.arctan2(ty,tx)
    d=np.stack([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)],1)
    # p as function of cos(theta_n) only (axial) -> evaluate along n in xz plane averaged over azimuth = use n=(sin,0,cos) and average over d's azimuth already isotropic
    ct=x; st=np.sqrt(1-ct**2)
    n=np.stack([st,0*st,ct],1)
    P=np.array([np.mean(np.exp(a2*eval_legendre(2,d@nn))) for nn in n])
    E=np.log(P)
    return [ (2*l+1)/2*np.sum(w*E*eval_legendre(l,ct)) for l in range(0,lmax+1)]
for a2,sig2 in [(2.0,0.01),(2.0,0.02),(4.0,0.01)]:
    c=a_coeffs(a2,sig2)
    print("HK a2=%.1f sig2=%.3f : a2'=%.4f a4=%.5f  pred a4=%.5f  a6=%.6f"%(a2,sig2,c[2],c[4],-18/35*a2**2*sig2,c[6]))
