import numpy as np, glob
from numpy.polynomial.legendre import leggauss
x,w=leggauss(400); P2=lambda c:1.5*c**2-.5
oracle_a2={0.80:5.51,0.95:3.97,1.05:2.92,1.10:2.16,1.12:1.67}
def Idir(a2):  # Fisher info per site per tilt dimension for p ∝ exp(a2 P2(n.d))
    p=np.exp(a2*P2(x)); p/=np.sum(w*p)
    return 4.5*a2**2*np.sum(w*p*x**2*(1-x**2))
def frame(S):
    Q=np.einsum('cxyzi,cxyzj->cij',S,S)/S[0,...,0].size; return np.linalg.eigh(Q)[1][:,:,-1]
def cell_spread(S,dg,b):
    c,L=S.shape[0],S.shape[1]; Sb=S.reshape(c,L//b,b,L//b,b,L//b,b,3)
    Qb=np.einsum('cxaybzdi,cxaybzdj->cxyzij',Sb,Sb); db=np.linalg.eigh(Qb)[1][...,-1]
    cos=np.abs(np.einsum('cxyzi,ci->cxyz',db,dg)); return np.mean(np.arccos(np.clip(cos,0,1))**2)
rng=np.random.default_rng(0); N=12**3
print("R=1 (global): KL_opt ~ sum over 2 director dims of 1/2 ln(N*I) (+ S dim, + O(1))")
for T,a2 in oracle_a2.items():
    I=Idir(a2); print(f"  T={T:.2f} a2={a2:.2f}  I/site/dim={I:.2f}  director part={np.log(N*I):.1f} nats  (x~1.5 with S: ~{1.5*np.log(N*I):.0f})")
print("\nR>1 cells: sigma_cell^2 = spread(actual) - spread(shuffled); G = n*I*sigma^2 (2 dims); net/cell = G - ln(1+G)")
print(" T    R  n   spread  noise   sigma2   G      net/cell  cells  net/config  KL_extra")
for f in sorted(glob.glob('ll_L12_T*_o.npy')):
    T=float(f.split('_T')[1][:5]);
    if T not in oracle_a2: continue
    S=np.load(f).astype(float)[:100]; dg=frame(S); I=Idir(oracle_a2[T])
    Ssh=S.reshape(len(S),-1,3).copy()
    for k in range(len(S)): Ssh[k]=Ssh[k][rng.permutation(N)]
    Ssh=Ssh.reshape(S.shape)
    for R,b in [(2,6),(3,4),(4,3)]:
        a=cell_spread(S,dg,b); nz=cell_spread(Ssh,dg,b); s2=max(a-nz,0); n=b**3
        G=n*I*s2/1.0  # 2 dims each with variance s2/2 -> sum = I*s2
        net=G-np.log1p(G); cells=R**3
        print(f" {T:.2f} {R}  {n:3d}  {a:.4f}  {nz:.4f}  {s2:.4f}  {G:6.2f}  {net:6.2f}    {cells:3d}   {cells*net:7.1f}   {cells*np.log1p(G):6.1f}")
