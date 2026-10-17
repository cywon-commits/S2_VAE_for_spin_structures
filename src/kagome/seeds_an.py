import numpy as np, glob, re
from oracle_kag import analyze
import collections
D=collections.defaultdict(list)
for f in glob.glob('seeds/kag_L12_T*_s*.npy'):
    T=float(re.search(r'T([\d.]+)_s',f).group(1)); D[T].append(analyze(np.load(f)))
print(" T      | lam_min        |psi3|        | -ln m3 (=4.5<s2>)  ln m6/ln m3 (uniform=4) | <eps^2>          | cumulant fit: mu=<s2_par>  sd(s2)/mu | m9 pred/obs")
out=[]
for T in sorted(D):
    R=D[T]; g=lambda k: (np.mean([r[k] for r in R]), np.std([r[k] for r in R])/np.sqrt(len(R)))
    lam,octp,eps=g('lam'),g('oct'),g('eps2')
    m3=np.array([r['m3'] for r in R]); m6=np.array([r['m6'] for r in R]); m9=np.array([r['m9'] for r in R])
    # pooled over seeds: average moments (mixture over seeds is part of physics: different colorings)
    M3,M6,M9=m3.mean(),m6.mean(),m9.mean()
    a,b=np.log(M3),np.log(M6)
    v=(b-4*a)/(162-40.5); mu=(10.125*v-a)/4.5
    m9p=np.exp(-40.5*mu+0.5*40.5**2*v)
    out.append((T,mu,eps[0],lam[0]))
    print(f" {T:.4f} | {lam[0]:.3f}±{lam[1]:.3f} | {octp[0]:.3f}±{octp[1]:.3f} | {-a:.4f}             {b/a:5.2f}                     | {eps[0]:.4f}±{eps[1]:.4f} | {mu:.4f}   {np.sqrt(max(v,0))/mu:5.2f}            | {m9p:.3f}/{M9:.3f}")
o=np.array(out)
for name,col in [("<s2_par> (cumulant mu)",1),("<eps^2>",2)]:
    sl=np.polyfit(np.log(o[:4,0]),np.log(o[:4,col]),1)[0]; print(f"log-log slope T<=0.004 for {name}: {sl:.2f}")
