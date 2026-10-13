import numpy as np
from fitlib import Fitter
rng=np.random.default_rng(9); N=600000
def kagome_caps(sp,so,n):
    k=rng.integers(0,3,n); phi=2*np.pi*k/3+rng.normal(scale=sp,size=n); eps=rng.normal(scale=so,size=n)
    return np.stack([np.cos(eps)*np.cos(phi),np.cos(eps)*np.sin(phi),np.sin(eps)],1)
sp,so=0.15,0.35
S=kagome_caps(sp,so,N); phi=np.arctan2(S[:,1],S[:,0])
m3=np.mean(np.cos(3*phi)); m6=np.mean(np.cos(6*phi))
print(f"moments: <cos3phi>={m3:.4f} (pred e^(-9s^2/2)={np.exp(-4.5*sp**2):.4f}), <cos6phi>={m6:.4f}, <cos3phi>^4={m3**4:.4f}")
print(f"         <P2(n_z)>+1/2={np.mean(1.5*S[:,2]**2):.4f}  (pred 3/2 s_perp^2 ~ {1.5*so**2:.4f}, exact 3/2<sin^2 eps>={1.5*np.mean(np.sin(rng.normal(scale=so,size=N))**2):.4f})")
print("NLL vs L (saturation order):")
nlls={}
for L in [1,2,3,4,6,8,10,12]:
    nlls[L]=Fitter(L,alpha=1e-7).fit(S).nll
for L,v in nlls.items(): print(f"  L={L:2d}: excess={v-nlls[12]:+.4f}")
print("equatorial Fourier of fitted E (off-support cusp test), L=12:")
ph=np.linspace(0,2*np.pi,720,endpoint=False); Xeq=np.stack([np.cos(ph),np.sin(ph),0*ph],1)
for a in [1e-7,1e-5,1e-4,1e-3]:
    f=Fitter(12,alpha=a).fit(S); E=f.E(Xeq)
    c3=-2*np.mean(E*np.cos(3*ph)); c6=-2*np.mean(E*np.cos(6*ph))
    print(f"  alpha={a:.0e}: c6/c3={c6/c3:+.3f}  (cusp prediction -0.25)   c3*sigma_par^2={c3*sp**2:.3f} (pred 2/9={2/9:.3f})")
