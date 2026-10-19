import numpy as np
from scipy.special import sph_harm_y, eval_legendre
np.set_printoptions(precision=4, suppress=True)
rng=np.random.default_rng(0)
# quadrature on sphere: fine grid
nt,npp=200,400
tg=np.arccos(np.polynomial.legendre.leggauss(nt)[0]); wt=np.polynomial.legendre.leggauss(nt)[1]
pg=np.linspace(0,2*np.pi,npp,endpoint=False)
TH,PH=np.meshgrid(tg,pg,indexing='ij'); W=np.outer(wt,np.ones(npp))*(2*np.pi/npp)
N=np.stack([np.sin(TH)*np.cos(PH),np.sin(TH)*np.sin(PH),np.cos(TH)],-1)
def proj(F,l,m): return np.sum(W*F*np.conj(sph_harm_y(l,m,TH,PH)))
# ---------- 3. pyrochlore: G_l(n) = sum_{beta != alpha} P_l(n.e_beta), in e_alpha frame ----------
E=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])/np.sqrt(3)
ea=E[0]; z=ea; xa=np.cross([0,0,1.0],z); 
# pick x axis along projection of e_beta=E[1] onto plane perp to z
xa=E[1]-(E[1]@z)*z; xa/=np.linalg.norm(xa); ya=np.cross(z,xa)
Rm=np.stack([xa,ya,z])   # rows: local axes
Eloc=E@Rm.T
for l in [2,4,6]:
    G=sum(eval_legendre(l,N@Eloc[b]) for b in [1,2,3])
    c={m:proj(G,l,m) for m in range(-l,l+1)}
    pw={abs(m):0 for m in c}
    for m,v in c.items(): pw[abs(m)]+=abs(v)**2
    print("l=%d power by |m|:"%l,{k:round(v,5) for k,v in pw.items() if v>1e-8})
# ---------- 4. triple-q skyrmion lattice moments ----------
def texture(m0,kind,phis=(np.pi,np.pi,np.pi),Ngrid=240):
    q=1.0
    qs=[q*np.array([np.cos(2*np.pi*a/3),np.sin(2*np.pi*a/3)]) for a in range(3)]
    # unit cell of triangular lattice of skyrmions: lattice vectors a1,a2 with q.a = 2pi integer
    a1=np.array([4*np.pi/np.sqrt(3),0]); a1=np.array([2*np.pi/q*2/np.sqrt(3)*np.cos(np.pi/6),2*np.pi/q*2/np.sqrt(3)*np.sin(np.pi/6)])
    # build reciprocal: choose a1,a2 so that q_a . a_i in 2pi Z
    Q=np.array(qs[:2]); A=2*np.pi*np.linalg.inv(Q).T  # rows a1,a2 with q_i.a_j=2pi delta
    u=np.linspace(0,1,Ngrid,endpoint=False); U,V=np.meshgrid(u,u,indexing='ij')
    R=U[...,None]*A[0]+V[...,None]*A[1]
    m=np.zeros(R.shape[:2]+(3,)); m[...,2]+=m0
    for a in range(3):
        qh=qs[a]/np.linalg.norm(qs[a]); ph=R@qs[a]+phis[a]
        if kind=='bloch': e=np.array([-qh[1],qh[0],0])
        else: e=np.array([qh[0],qh[1],0])
        m+= np.cos(ph)[...,None]*np.array([0,0,1.0]) + np.sin(ph)[...,None]*e
    m/=np.linalg.norm(m,axis=-1)[...,None]
    # topological charge
    du=np.gradient(m,axis=0); dv=np.gradient(m,axis=1)
    Qtop=np.sum(np.einsum('ijk,ijk->ij',m,np.cross(du,dv)))/(4*np.pi)
    return m.reshape(-1,3),Qtop
def moments(mvec,l,m):
    th=np.arccos(np.clip(mvec[:,2],-1,1)); ph=np.arctan2(mvec[:,1],mvec[:,0])
    return np.mean(sph_harm_y(l,m,th,ph))
for kind in ['bloch','neel']:
    for m0 in [0.0,0.5,1.0]:
        mv,Qt=texture(m0,kind)
        out=[]
        bad=0
        for l in range(1,8):
            for mm in range(-l,l+1):
                v=moments(mv,l,mm)
                if mm%6!=0 and abs(v)>1e-3: bad+=1
        y66=moments(mv,6,6); y10=moments(mv,1,0); y20=moments(mv,2,0)
        print(f"{kind:5s} m0={m0:.1f} Q={Qt:+.3f} <Y10>={y10.real:+.3f} <Y20>={y20.real:+.3f} <Y66>={y66:.4f}  forbidden_nonzero={bad}")
