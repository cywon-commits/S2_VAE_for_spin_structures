import numpy as np, importlib.util, sys
spec=importlib.util.spec_from_file_location("hm","harmonic.py"); 
src=open(__file__.replace("flatband.py","harmonic.py")).read().split("L=int(sys.argv")[0]; exec(src)
for L in (6,9,12):
    bonds,N=lattice(L); out=[]
    for hm in (0.1,0.5,1.0):
        S,h=q0_state("umbrella",hm,N); w,_=spectrum(S,h,bonds,N); w=np.sort(w)
        Sc,_=q0_state("coplanar",hm,N); wc,_=spectrum(Sc,h,bonds,N); wc=np.sort(wc)
        nzc=(wc<1e-7).sum(); nzu=(w<1e-7).sum()
        lifted=w[nzu:nzc]            # modes zero in coplanar but lifted in umbrella
        out.append((hm,nzc,nzu,lifted.mean()/hm**2,np.exp(np.log(lifted).mean())/hm**2, lifted.max()/hm**2))
    for o in out: print(f"L={L} N={N} h={o[0]}: zero(coplanar)={o[1]} zero(umbrella)={o[2]}  lifted modes: arith c={o[3]:.4f}  geom c={o[4]:.4f}  max c={o[5]:.4f}")
