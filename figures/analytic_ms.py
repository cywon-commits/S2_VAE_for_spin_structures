"""Mean-field Maier-Saupe S(T) curve (z=6, J=1): from src/analytic_ms_heatkernel.py."""
import numpy as np
from scipy.special import eval_legendre
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
x, w = leggauss(400)
def Zf(a): return np.sum(w * np.exp(a * eval_legendre(2, x))) / 2
def Sav(a): return np.sum(w * eval_legendre(2, x) * np.exp(a * eval_legendre(2, x))) / 2 / Zf(a)
def Ssol(t):
    g = lambda S: Sav(S / t) - S; Ss = np.linspace(1e-3, 0.999, 1500); vals = [g(s) for s in Ss]; roots = [brentq(g, Ss[i], Ss[i + 1]) for i in range(len(Ss) - 1) if vals[i] * vals[i + 1] < 0]; return max(roots) if roots else 0.0
def ms_curve(z=6):
    ts = np.linspace(0.12, 0.2202, 40); S = np.array([Ssol(t) for t in ts]); return z * ts, S
