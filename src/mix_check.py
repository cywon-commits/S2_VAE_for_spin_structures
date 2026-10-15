import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre as Pl
from moment_est import est, M
x,w=leggauss(400)
def moms(a,s2):  # exact moments of exp(aP2) smeared by s2
    return np.exp(-1.5*s2)*M(a,2), np.exp(-5*s2)*M(a,4)
rng=np.random.default_rng(0)
print("posterior samples differ in amplitude a (mean a0, rel. spread r) and share smearing s2_true")
print(" a0   s2_true  r    | per-sample est (mean) | averaged-moment est")
for a0 in [4.0,3.0]:
  for s2 in [0.03,0.01]:
    for r in [0.05,0.10,0.20]:
        A=a0*(1+r*rng.normal(size=400))
        ms=np.array([moms(a,s2) for a in A])
        per=np.mean([est(*m)[1] for m in ms])
        avg=est(*ms.mean(0))[1]
        print(f" {a0:.1f}  {s2:.3f}   {r:.2f} |       {per:+.4f}         |     {avg:+.4f}")
