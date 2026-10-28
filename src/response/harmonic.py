"""Harmonic spectrum of kagome AF (J=1) in field h for q=0 ground states: coplanar 'Y' (h in plane) vs umbrella (h normal to plane).
E = sum_bonds S_i.S_j - h.sum S_i.  Tangent-space Hessian: E2 = 1/2 u.A.u - 1/2 sum_i (f_i.S_i)|u_i|^2, f_i = h - sum_nb S_j."""
import numpy as np, sys
def lattice(L):
    idx=lambda x,y,s: ((x%L)*L+(y%L))*3+s
    NB={0:[(1,0,0),(1,-1,0),(2,0,0),(2,0,-1)],1:[(0,0,0),(0,1,0),(2,0,0),(2,1,-1)],2:[(0,0,0),(0,0,1),(1,0,0),(1,-1,1)]}
    bonds=set()
    for x in range(L):
        for y in range(L):
            for s in range(3):
                for t,dx,dy in NB[s]:
                    i,j=idx(x,y,s),idx(x+dx,y+dy,t); bonds.add((min(i,j),max(i,j)))
    return sorted(bonds), 3*L*L
def spectrum(S, h, bonds, N):
    A=np.zeros((3*N,3*N))
    for i,j in bonds:
        A[3*i:3*i+3,3*j:3*j+3]+=np.eye(3); A[3*j:3*j+3,3*i:3*i+3]+=np.eye(3)
    f=np.tile(h,(N,1)); 
    for i,j in bonds: f[i]-=S[j]; f[j]-=S[i]
    par=np.einsum('ij,ij->i',f,S); perp=np.linalg.norm(f-par[:,None]*S,axis=1).max()
    P=np.zeros((3*N,2*N))
    for i in range(N):
        t=np.array([1.,0,0]) if abs(S[i,0])<0.9 else np.array([0,1.,0])
        e1=np.cross(S[i],t); e1/=np.linalg.norm(e1); e2=np.cross(S[i],e1)
        P[3*i:3*i+3,2*i]=e1; P[3*i:3*i+3,2*i+1]=e2
    H=P.T@A@P + np.diag(np.repeat(par,2))
    return np.linalg.eigvalsh(H), perp
def q0_state(kind, hm, N):
    z=np.array([0,0,1.]); b=hm/2      # S_triangle = h/(2J)
    if kind=="umbrella":
        c=b/3; s=np.sqrt(1-c*c); ang=np.array([0,2,4])*np.pi/3
        sub=np.stack([s*np.cos(ang),s*np.sin(ang),c*np.ones(3)],1)
    else:  # coplanar Y in x-z plane: S_A along +z? solve S_A=(0,0,1)?? use symmetric Y: S_A=z, S_B,S_C at angles +-phi
        # S_A + S_B + S_C = b z with S_A=(0,0,1): cos(phi)=(b-1)/2  (valid for b<=3 ... b in [-1,3])
        cphi=(b-1)/2; sphi=np.sqrt(1-cphi**2)
        sub=np.array([[0,0,1.],[sphi,0,cphi],[-sphi,0,cphi]])
    return np.tile(sub,(N//3,1)), hm*z
L=int(sys.argv[1]) if len(sys.argv)>1 else 6
bonds,N=lattice(L)
print(f"N={N}; zero modes counted as eig<1e-7; 'soft' = number of eigenvalues below 0.05")
for hm in [0.0,0.1,0.3,0.5,1.0,1.5,2.0]:
    row=[]
    for kind in ("coplanar","umbrella"):
        S,h=q0_state(kind,hm,N); w,perp=spectrum(S,h,bonds,N)
        nz=(w<1e-7).sum(); nneg=(w<-1e-7).sum()
        # lowest nonzero band of the former flat modes: eigenvalue at index N/3
        flat=np.sort(w)[N//3-1]
        row.append(f"{kind}: zero={nz:3d} neg={nneg} flat-band top={flat:.4f} (pred h^2/... ) resid={perp:.1e}")
    print(f"h={hm:4.2f}  "+" | ".join(row))
