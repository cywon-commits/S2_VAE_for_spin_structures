"""Moment estimator: <P_l>_obs = exp(-l(l+1) s2/4) M_l(a_loc), l=2,4 -> (a_loc, s2).  Vectorised via a lookup table."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre as Pl
_x, _w = leggauss(400)
_A = np.concatenate([np.linspace(0.05, 5, 2000), np.linspace(5, 60, 3000)[1:]])
def _M(a, l):
    p = np.exp(np.outer(a, Pl(2, _x)) - (a * 1.0)[:, None]); return (p * Pl(l, _x) * _w).sum(1) / (p * _w).sum(1)
_M2, _M4 = _M(_A, 2), _M(_A, 4)
_H = np.log(_M4) - (10 / 3) * np.log(_M2)              # h(a) = ln M4 - (10/3) ln M2 ; solved against ln m4 - (10/3) ln m2
def est_vec(m2, m4):
    """m2, m4 arrays -> (a_loc, sigma2); NaN where no solution (m2<=0, m4<=0, outside table)."""
    m2 = np.asarray(m2, float); m4 = np.asarray(m4, float); out_a = np.full(m2.shape, np.nan); out_s = np.full(m2.shape, np.nan)
    ok = (m2 > 0) & (m4 > 0)
    t = np.log(m4[ok]) - (10 / 3) * np.log(m2[ok])
    # h is monotone decreasing in a on the table -> invert by interpolation
    order = np.argsort(_H); a = np.interp(t, _H[order], _A[order], left=np.nan, right=np.nan)
    M2a = np.interp(a, _A, _M2); s = (np.log(M2a) - np.log(m2[ok])) / 1.5
    out_a[ok], out_s[ok] = a, s; return out_a, out_s
