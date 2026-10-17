import numpy as np, glob, re, collections
from block_defects import classify_blocks
def cell_frames(S,b=3):
    L=S.shape[0]; c=S.reshape(L//b,b,L//b,b,3,3).transpose(0,2,1,3,4,5).reshape(L//b,L//b,-1,3)
    Q=np.einsum('xyni,xynj->xyij',c,c); V=np.linalg.eigh(Q)[1]; nu=V[...,0]; e1=V[...,2]; e2=np.cross(nu,e1)
    z=np.einsum('xyni,xyi->xyn',c,nu); phi=np.arctan2(np.einsum('xyni,xyi->xyn',c,e2),np.einsum('xyni,xyi->xyn',c,e1))
    psi=np.mean(np.exp(3j*phi),-1); return z,phi-np.angle(psi)[...,None]/3
D=collections.defaultdict(lambda: {'z':[],'p':[],'zc':[],'pc':[],'frac':[]})
for f in glob.glob('seeds/kag_L12_T*_s*.npy'):
    T=float(re.search(r'T([\d.]+)_s',f).group(1))
    for S in np.load(f).astype(float)[::4]:
        z,p=cell_frames(S,3); bad_loop=(classify_blocks(S,3)!='none')
        bad=bad_loop|np.roll(bad_loop,1,0)|np.roll(bad_loop,1,1)|np.roll(np.roll(bad_loop,1,0),1,1)  # 4 blocks around each loop
        ok=~bad; D[T]['frac'].append(ok.mean())
        D[T]['zc'].append(z[ok].ravel()); D[T]['pc'].append(p[ok].ravel()); D[T]['z'].append(z.ravel()); D[T]['p'].append(p.ravel())
rows=[]
print(" T      | clean cell frac | ALL cells: <eps^2>  -ln m3  r6   r9 | DEFECT-FREE cells: <eps^2>  -ln m3  r6   r9")
for T in sorted(D):
    def st(Z,P):
        Z=np.concatenate(Z); P=np.concatenate(P); e=np.mean(np.arcsin(np.clip(Z,-1,1))**2)
        m3,m6,m9=[np.mean(np.cos(k*P)) for k in (3,6,9)]; return e,-np.log(m3),np.log(m6)/np.log(m3),np.log(m9)/np.log(m3)
    a=st(D[T]['z'],D[T]['p']); c=st(D[T]['zc'],D[T]['pc']); rows.append((T,a,c))
    print(f" {T:.4f} | {np.mean(D[T]['frac']):.2f}            | {a[0]:.4f} {a[1]:.4f} {a[2]:.2f} {a[3]:.2f} | {c[0]:.4f} {c[1]:.4f} {c[2]:.2f} {c[3]:.2f}")
Ts=np.array([r[0] for r in rows])
for lab,j in [("ALL",1),("DEFECT-FREE",2)]:
    e=np.array([r[j][0] for r in rows]); m=np.array([r[j][1] for r in rows]); sel=Ts<=0.0041
    print(f"{lab:12s}: slope in-plane (-ln m3) = {np.polyfit(np.log(Ts[sel]),np.log(m[sel]),1)[0]:.2f},  slope out-of-plane <eps^2> = {np.polyfit(np.log(Ts[sel]),np.log(e[sel]),1)[0]:.2f}")
