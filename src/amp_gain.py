"""Information in per-cell amplitude (local S) beyond cell director. Uniaxial P2 family, per-cell MLE a_c vs pooled a.
Overfit removed with site-shuffled control. Budget: 1 dim, net = G_a - 1/2 ln(1+2 G_a)."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import interp1d
x,w=leggauss(300); P2x=1.5*x**2-.5
A=np.linspace(-2,40,4000); Z=np.array([np.sum(w*np.exp(a*P2x)) for a in A]); M2=np.array([np.sum(w*P2x*np.exp(a*P2x)) for a in A])/Z
a_of_m=interp1d(M2,A,bounds_error=False,fill_value=(A[0],A[-1])); logZ=interp1d(A,np.log(Z))
def cell_m2(S,b):
    c,L=S.shape[0],S.shape[1]; Sb=S.reshape(c,L//b,b,L//b,b,L//b,b,3)
    Qb=np.einsum('cxaybzdi,cxaybzdj->cxyzij',Sb,Sb); d=np.linalg.eigh(Qb)[1][...,-1]
    cs=np.einsum('cxaybzdi,cxyzi->cxyzabd',Sb,d); return (1.5*cs**2-.5).mean(axis=(-3,-2,-1)).ravel()
def nll_gain(m2c):
    ap=float(a_of_m(m2c.mean())); pooled=-(ap*m2c.mean()-logZ(ap))
    ac=a_of_m(m2c); per=-np.mean(ac*m2c-logZ(ac)); return pooled-per
rng=np.random.default_rng(2)
oracle_dir={0.95:(0.033,0.088,0.145),1.05:(0.034,0.087,0.146)}   # v4 cell_gain2 values
vae={0.95:(0.040,0.098,0.156,48,127,243),1.05:(0.042,0.105,0.176,48,130,252)}
budget_dir={0.95:(19,53,102),1.05:(17,51,100)}
N=1728
for T in (0.95,1.05):
    S=np.load(f'll_L12_T{T:.3f}_o.npy').astype(float)[:100]
    Ssh=S.reshape(len(S),-1,3).copy()
    for k in range(len(S)): Ssh[k]=Ssh[k][rng.permutation(N)]
    Ssh=Ssh.reshape(S.shape)
    print(f"T={T}:  R | amp gain (net/site) | G_a/cell  amp KL budget | dir+amp gain vs VAE gain | dir+amp KL budget vs VAE extra KL")
    for k,(R,b) in enumerate([(2,6),(3,4),(4,3)]):
        g=nll_gain(cell_m2(S,b))-nll_gain(cell_m2(Ssh,b)); n=b**3; cells=R**3
        Ga=g*n; kla=cells*0.5*np.log1p(2*Ga)
        tot=oracle_dir[T][k]+g; klt=budget_dir[T][k]+kla
        print(f"        {R} |      {g:.4f}          |  {Ga:5.2f}      {kla:5.1f}      |   {tot:.3f} vs {vae[T][k]:.3f}        |   {klt:5.0f} vs {vae[T][k+3]}")
