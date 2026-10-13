import numpy as np
from samplers import sample_x_quadratic, assemble
from fitlib import Fitter
rng=np.random.default_rng(5); N=600000; z=np.array([0,0,1.])
c=1/(2*0.15**2*np.sin(np.pi/3)**2); x=sample_x_quadratic(-c,-0.5,N,rng); S=assemble(x,z,rng)
ref=Fitter(2,alpha=0).fit(S).nll
print("ring, L=8, alpha sweep  (exact family is l<=2)")
for a in [1e-7,1e-5,1e-4,1e-3,1e-2]:
    f=Fitter(8,alpha=a).fit(S); P=f.power(); tot=sum(P.values())
    frac=sum(P[l] for l in (1,2))/tot
    print(f" alpha={a:.0e}: NLL excess={f.nll-ref:+.4f}  power fraction in l<=2 = {frac:.3f}  P_l:",
          " ".join(f"{l}:{P[l]:.1f}" for l in range(1,9)))
# also the Z2 caps with L=8: odd power vs alpha
from samplers import sample_linear
S2=sample_linear(10*z,N,rng); S2*=rng.choice([-1,1],N)[:,None]
print("\nZ2 caps, L=8: odd/even power ratio vs alpha")
for a in [1e-7,1e-5,1e-4,1e-3]:
    f=Fitter(8,alpha=a).fit(S2); P=f.power()
    print(f" alpha={a:.0e}: odd/even={sum(P[l] for l in P if l%2)/sum(P[l] for l in P if l%2==0):.1e}")
