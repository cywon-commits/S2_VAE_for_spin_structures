"""Shim: the two helpers of kagome/loops.py without its module-level analysis loop (which runs on import)."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kagome"))
import numpy as np
from kagome_mc import NB
def build_nbrs(L):
    nb=np.zeros((L,L,3,4),int)
    idx=lambda x,y,s: ((x%L)*L+(y%L))*3+s
    for x in range(L):
        for y in range(L):
            for s in range(3):
                nb[x,y,s]=[idx(x+dx,y+dy,t) for t,dx,dy in NB[s]]
    return nb.reshape(-1,4)
def global_triad(S):
    X=S.reshape(-1,3); V=np.linalg.eigh(X.T@X)[1]; nu,e1=V[:,0],V[:,2]; e2=np.cross(nu,e1)
    phi=np.arctan2(X@e2,X@e1); th=np.angle(np.mean(np.exp(3j*phi)))/3
    return np.array([np.cos(th+2*np.pi*k/3)*e1+np.sin(th+2*np.pi*k/3)*e2 for k in range(3)])
