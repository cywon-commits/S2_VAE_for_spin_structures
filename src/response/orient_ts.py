import numpy as np, sys, json
from kagome_mc_field import run
L,T,h,seed=12,float(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3])
O,N=run(L,T,[0,0,h],2000,int(sys.argv[4]),5,seed,every=20)
u2=O[:,3]; nb=5; B=np.array_split(u2,nb)
print(json.dumps(dict(T=T,h=h,seed=seed,u2=float(u2.mean()),blocks=[round(float(b.mean()),3) for b in B],oop=float(O[:,4].mean()))))
