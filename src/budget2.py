import numpy as np
from numpy.polynomial.legendre import leggauss
x,w=leggauss(400); P2=lambda c:1.5*c**2-.5
def Idir(a2):
    p=np.exp(a2*P2(x)); p/=np.sum(w*p); return 4.5*a2**2*np.sum(w*p*x**2*(1-x**2))
a2o={0.80:5.51,0.95:3.97,1.05:2.92,1.10:2.16}
s2cell={0.80:(0.0064,0.0149,0.0226),0.95:(0.0103,0.0223,0.0342),1.05:(0.0131,0.0337,0.0533),1.10:(0.0260,0.0644,0.1049)}
obs_gain={0.80:(0.037,0.086,0.127),0.95:(0.038,0.087,0.111),1.05:(0.041,0.100,0.146),1.10:(0.043,0.107,0.167)}
obs_kl={0.80:(96,283,559),0.95:(112,306,553),1.05:(87,247,466),1.10:(75,205,383)}
kl1={0.80:21,0.95:28,1.05:20,1.10:17}
N=1728
print(" T    R | NLL gain: old pred  corrected  observed | extra KL: hier(dev) indep(full) observed | net ELBO hier-prior (corrected)")
for T in a2o:
    I=Idir(a2o[T])
    for k,(R,n) in enumerate([(2,216),(3,64),(4,27)]):
        s2=s2cell[T][k]; Gold=n*I*s2; G=n*I*s2/2; cells=R**3
        gain_old=cells*Gold/N; gain=cells*G/N
        kl_h=cells*np.log1p(G)
        # independent per-cell prior: each cell re-encodes director (2 dims) + S: scale R=1 KL by ln(nI)/ln(NI)
        kl_i=cells*kl1[T]*np.log(n*I)/np.log(N*I)-kl1[T]
        net=cells*(G-np.log1p(G))
        print(f" {T:.2f} {R} |   {gain_old:.3f}     {gain:.3f}     {obs_gain[T][k]:.3f}  |    {kl_h:5.0f}      {kl_i:5.0f}     {obs_kl[T][k]:4d}  |   {net:+6.0f}")
