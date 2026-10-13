import numpy as np
from samplers import sample_linear, sample_x_quadratic, assemble
from fitlib import Fitter
rng=np.random.default_rng(5); N=600000
z=np.array([0,0,1.])
def table(name,S,Ls=(1,2,4,8)):
    out=[]
    for L in Ls:
        f=Fitter(L,alpha=1e-7).fit(S); P=f.power()
        odd=sum(v for l,v in P.items() if l%2); ev=sum(v for l,v in P.items() if l%2==0)
        out.append((L,f.nll,odd,ev,f))
    print(name); best=out[-1][1]
    for L,nll,odd,ev,_ in out: print(f"   L={L}: NLL={nll:.4f} (excess {nll-best:+.4f} nat)  odd power={odd:.2e} even power={ev:.2e}")
    return out
# T2: Z2 gauge -> two caps
S=sample_linear(10*z,N,rng); S*=rng.choice([-1,1],N)[:,None]
table("T2  Z2: +/- caps (kappa=10)",S)
# T3: U(1) ring: theta0=60 deg, width ~0.15 rad, azimuth uniform  (p(x) ∝ exp(-c (x-cos60)^2))
c=1/(2*0.15**2*np.sin(np.pi/3)**2); x=sample_x_quadratic(-c,-0.5,N,rng); S=assemble(x,z,rng)
out=table("T3  U(1): ring at theta0=60deg",S)
f=out[-1][4]; print("   ring L=8 axial coefs a_l (P_l units):",[round(f.coef(l,0)*np.sqrt((2*l+1)/(4*np.pi)),3) for l in range(1,9)])
