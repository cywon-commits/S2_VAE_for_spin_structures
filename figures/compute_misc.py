"""Analytic / toy computations behind Figs 2, 3, S1-S6 (chat scripts logz.py, toy1.py, toy3.py, toy4.py, sym_check.py, analytic_pyrochlore_skl.py, test_samplers.py, acc.py of the zip,
refactored to return numbers instead of printing; fixed seeds).  Output: figures/data/misc.json"""
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, f"{ROOT}/src")
from scipy.integrate import lebedev_rule, quad
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation as Rot
from scipy.special import sph_harm_y, eval_legendre
from numpy.polynomial.legendre import leggauss
from samplers import sample_linear, sample_x_quadratic, sample_bingham, assemble
from fitlib import Fitter
out = {}; todo = sys.argv[1:] or ["logz", "toy3", "toy1", "sym", "pyro", "skl", "toy4", "samplers"]
def save(): json.dump(out, open(f"{HERE}/data/misc.json", "w"), indent=1, default=float)
if "logz" in todo:
    def lz_leb(f, order): X, W = lebedev_rule(order); v = f(X.T); m = v.max(); return m + np.log(np.sum(W * np.exp(v - m)))
    exact = lambda k: np.log(4 * np.pi * np.sinh(k) / k) if k < 300 else k + np.log(2 * np.pi / k); axis = np.array([0.3, -0.5, 0.81]); axis /= np.linalg.norm(axis)
    ks = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 1000]; orders = [17, 35, 59, 131]
    out["logz_err"] = dict(kappa=ks, orders=orders, err={str(o): [abs(lz_leb(lambda X: k * (X @ axis), o) - exact(k)) for k in ks] for o in orders})
    allo = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131]
    kk = [1, 2, 5, 10, 20, 50, 100, 200]; need = []
    for k in kk:
        ok = [o for o in allo if abs(lz_leb(lambda X: k * (X @ axis), o) - exact(k)) < 1e-3]; need.append(ok[0] if ok else None)
    out["logz_need"] = dict(kappa=kk, degree=need); print("logz done", flush=True); save()
if "toy3" in todo:
    rng = np.random.default_rng(5); N = 300000; z = np.array([0, 0, 1.]); c = 1 / (2 * 0.15 ** 2 * np.sin(np.pi / 3) ** 2); x = sample_x_quadratic(-c, -0.5, N, rng); S = assemble(x, z, rng); ref = Fitter(2, alpha=0).fit(S).nll; rows = []
    for a in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        f = Fitter(8, alpha=a).fit(S); P = f.power(); tot = sum(P.values()); rows.append(dict(alpha=a, nll_excess=f.nll - ref, P={str(l): P[l] / tot for l in P}))
    out["toy3"] = rows; print("toy3 done", flush=True); save()
if "toy1" in todo:
    from numpy.polynomial.legendre import leggauss as lg
    xg, wg = lg(400)
    def M(a, l): p = np.exp(a * eval_legendre(2, xg)); return np.sum(wg * p * eval_legendre(l, xg)) / np.sum(wg * p)
    from scipy.optimize import fsolve
    def est(m2, m4): f = lambda v: [np.exp(-1.5 * v[1]) * M(v[0], 2) - m2, np.exp(-5 * v[1]) * M(v[0], 4) - m4]; a, s2 = fsolve(f, [4.0, 0.03]); return float(a), float(s2)
    rng = np.random.default_rng(11); N = 400000; rows = []
    for a2 in (2.0, 4.0, 8.0, 15.0):
        for s2 in (0.01, 0.04):
            base = sample_bingham(np.diag([0, 0, 1.5 * a2]), N, rng); tx, ty = rng.normal(scale=np.sqrt(s2 / 2), size=(2, N)); th = np.sqrt(tx ** 2 + ty ** 2); ph = np.arctan2(ty, tx)
            D = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1); z = np.array([0, 0, 1.]); ax = np.cross(np.tile(z, (N, 1)), D); s = np.linalg.norm(ax, axis=1); cc = D[:, 2]; ax = np.where(s[:, None] > 1e-12, ax / np.maximum(s, 1e-12)[:, None], 0)
            S = base * cc[:, None] + np.cross(ax, base) * s[:, None] + ax * (np.sum(ax * base, 1) * (1 - cc))[:, None]
            f = Fitter(8, alpha=1e-7).fit(S); a2f = f.coef(2, 0) * np.sqrt(5 / (4 * np.pi)); a4f = f.coef(4, 0) * np.sqrt(9 / (4 * np.pi)); m2 = np.mean(eval_legendre(2, S[:, 2])); m4 = np.mean(eval_legendre(4, S[:, 2])); al, se = est(m2, m4)
            rows.append(dict(a2=a2, s2_true=s2, a2_fit=a2f, a4_fit=a4f, s2_from_a4=-(35 / 18) * a4f / a2f ** 2, a_loc=al, s2_moment=se))
            print("toy1", rows[-1], flush=True)
    out["toy1"] = rows; save()
if "sym" in todo:
    rng = np.random.default_rng(0); P = rng.normal(size=(4000, 3)); P /= np.linalg.norm(P, axis=1)[:, None]
    def Ylm_real(l, pts):
        th = np.arccos(np.clip(pts[:, 2], -1, 1)); ph = np.arctan2(pts[:, 1], pts[:, 0]); o = []
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), th, ph); o.append(np.sqrt(2) * Y.imag if m < 0 else (Y.real if m == 0 else np.sqrt(2) * Y.real))
        return np.array(o).T
    Rz = lambda a: Rot.from_euler('z', a).as_matrix(); Rx = lambda a: Rot.from_euler('x', a).as_matrix(); I = -np.eye(3)
    def gen(gs):
        G = [np.eye(3)]; ch = True
        while ch:
            ch = False
            for g in list(G):
                for h in gs:
                    k = g @ h
                    if not any(np.allclose(k, x) for x in G): G.append(k); ch = True
        return G
    groups = {"D∞h": gen([Rz(2 * np.pi / 60), Rx(np.pi), I]), "D3h": gen([Rz(2 * np.pi / 3), Rx(np.pi), I @ Rz(np.pi)]), "D3d": gen([Rz(2 * np.pi / 3), Rx(np.pi), I]), "C3v": gen([Rz(2 * np.pi / 3), I @ Rx(np.pi)]), "C6v": gen([Rz(2 * np.pi / 6), I @ Rx(np.pi)]), "C∞v": gen([Rz(2 * np.pi / 60), I @ Rx(np.pi)])}
    cnt = {}
    for name, G in groups.items():
        cs = []
        for l in range(0, 7):
            Y = Ylm_real(l, P); Dv = np.zeros((2 * l + 1, 2 * l + 1))
            for g in G: Yg = Ylm_real(l, P @ g); D, *_ = np.linalg.lstsq(Y, Yg, rcond=None); Dv += D
            Dv /= len(G); cs.append(int(round(np.trace(Dv))))
        cnt[name] = cs
    out["sym"] = cnt; print("sym", cnt, flush=True); save()
if "pyro" in todo:
    nt, npp = 200, 400; xs, ws = leggauss(nt); tg = np.arccos(xs); pg = np.linspace(0, 2 * np.pi, npp, endpoint=False); TH, PH = np.meshgrid(tg, pg, indexing='ij'); W = np.outer(ws, np.ones(npp)) * (2 * np.pi / npp); N3 = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1)
    E = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3); z = E[0]; xa = E[1] - (E[1] @ z) * z; xa /= np.linalg.norm(xa); ya = np.cross(z, xa); Rm = np.stack([xa, ya, z]); Eloc = E @ Rm.T; res = {}
    for l in (2, 4, 6):
        G = sum(eval_legendre(l, N3 @ Eloc[b]) for b in (1, 2, 3)); pw = {}
        for m in range(-l, l + 1): v = np.sum(W * G * np.conj(sph_harm_y(l, m, TH, PH))); pw[abs(m)] = pw.get(abs(m), 0) + abs(v) ** 2
        res[str(l)] = {str(k): v for k, v in pw.items() if v > 1e-8}
    out["pyrochlore"] = res; print("pyro", res, flush=True); save()
if "skl" in todo:
    def texture(m0, kind, Ngrid=240):
        q = 1.0; qs = [q * np.array([np.cos(2 * np.pi * a / 3), np.sin(2 * np.pi * a / 3)]) for a in range(3)]; Q = np.array(qs[:2]); A = 2 * np.pi * np.linalg.inv(Q).T; u = np.linspace(0, 1, Ngrid, endpoint=False); U, V = np.meshgrid(u, u, indexing='ij'); R = U[..., None] * A[0] + V[..., None] * A[1]
        m = np.zeros(R.shape[:2] + (3,)); m[..., 2] += m0
        for a in range(3):
            qh = qs[a] / np.linalg.norm(qs[a]); ph = R @ qs[a] + np.pi; e = np.array([-qh[1], qh[0], 0]) if kind == 'bloch' else np.array([qh[0], qh[1], 0]); m += np.cos(ph)[..., None] * np.array([0, 0, 1.0]) + np.sin(ph)[..., None] * e
        m /= np.linalg.norm(m, axis=-1)[..., None]; du = np.gradient(m, axis=0); dv = np.gradient(m, axis=1); Qt = np.sum(np.einsum('ijk,ijk->ij', m, np.cross(du, dv))) / (4 * np.pi); return m.reshape(-1, 3), Qt
    def mom(mv, l, m): th = np.arccos(np.clip(mv[:, 2], -1, 1)); ph = np.arctan2(mv[:, 1], mv[:, 0]); return np.mean(sph_harm_y(l, m, th, ph))
    rows = []
    for kind in ("bloch", "neel"):
        for m0 in (0.0, 0.5, 1.0):
            mv, Qt = texture(m0, kind); moments = {f"{l},{mm}": complex(mom(mv, l, mm)) for l in range(1, 8) for mm in range(0, l + 1)}
            rows.append(dict(kind=kind, m0=m0, Q=Qt, Y10=moments["1,0"].real, Y20=moments["2,0"].real, Y66=moments["6,6"].real, Ylm_abs={k: abs(v) for k, v in moments.items()}))
    out["skl"] = rows; print("skl done", flush=True); save()
if "toy4" in todo:
    rng = np.random.default_rng(9); N = 400000
    def caps(sp, so, n): k = rng.integers(0, 3, n); phi = 2 * np.pi * k / 3 + rng.normal(scale=sp, size=n); eps = rng.normal(scale=so, size=n); return np.stack([np.cos(eps) * np.cos(phi), np.cos(eps) * np.sin(phi), np.sin(eps)], 1)
    sp, so = 0.15, 0.35; S = caps(sp, so, N); nll = {L: Fitter(L, alpha=1e-7).fit(S).nll for L in (1, 2, 3, 4, 6, 8, 10, 12)}; ph = np.linspace(0, 2 * np.pi, 720, endpoint=False); Xeq = np.stack([np.cos(ph), np.sin(ph), 0 * ph], 1); cusp = []
    for a in (1e-7, 1e-5, 1e-4, 1e-3):
        f = Fitter(12, alpha=a).fit(S); E = f.E(Xeq); c3 = -2 * np.mean(E * np.cos(3 * ph)); c6 = -2 * np.mean(E * np.cos(6 * ph)); cusp.append(dict(alpha=a, c6_over_c3=c6 / c3))
    out["toy4"] = dict(nll=nll, cusp=cusp, sigma_par=sp, sigma_perp=so); print("toy4 done", flush=True); save()
if "samplers" in todo:
    rng = np.random.default_rng(7); N = 400000; xg = np.linspace(-1, 1, 401); res = {}
    def hist(x, nb=60): h, e = np.histogram(x, bins=nb, range=(-1, 1), density=True); return dict(centers=(0.5 * (e[1:] + e[:-1])).tolist(), dens=h.tolist())
    def dens(logf): p = np.exp(logf(xg) - logf(xg).max()); return (p / np.trapezoid(p, xg)).tolist()
    for name, h in (("linear_k3", np.array([0, 0, 3.0])), ("linear_k20", np.array([0, 0, 20.0]))): S = sample_linear(h, N, rng); res[name] = dict(h=hist(S[:, 2]), exact=dens(lambda x, k=h[2]: k * x))
    for K, hz in ((-3.0, 1.0), (3.0, 1.0), (8.0, 0.5), (8.0, -12.0)): a = hz / (2 * K); x = sample_x_quadratic(K, a, N, rng); res[f"uni_K{K}_h{hz}"] = dict(h=hist(x), exact=dens(lambda t, K=K, hz=hz: K * t ** 2 + hz * t))
    A = np.diag([0, 0, 15.0]); S = sample_bingham(A, N, rng); res["bingham_a15"] = dict(h=hist(S[:, 2]), exact=dens(lambda t: 15 * t ** 2)); res["bingham_a15"]["note"] = "p(x) for the z component"
    def acc_quad(K, a):
        logf = lambda x: K * (x + a) ** 2; x0 = np.clip(-a, -1, 1); Z = quad(lambda x: np.exp(logf(x) - K * 4), -1, 1, points=[x0])[0]; env = 0
        for lo, hi in [(-1, x0), (x0, 1)]:
            if hi - lo < 1e-12: continue
            s = (logf(hi) - logf(lo)) / (hi - lo); c = logf(lo) - s * lo - K * 4; env += (np.exp(s * hi + c) - np.exp(s * lo + c)) / s if abs(s) > 1e-12 else np.exp(c) * (hi - lo)
        return Z / env
    Ks = np.logspace(0, 2, 20); res["acc_chord"] = {f"hz{hz}": [acc_quad(K, hz / (2 * K)) for K in Ks] for hz in (0, 5, -30)}; res["acc_chord_K"] = Ks.tolist()
    def acc_bing(lams, rng=np.random.default_rng(3)):
        lam = np.array(lams, float); lam = lam - lam.min(); b = brentq(lambda b: np.sum(1 / (b + 2 * lam)) - 1, 1e-10, 3.0); Om = 1 + 2 * lam / b; M = np.exp(-(3 - b) / 2) * (3 / b) ** 1.5; y = rng.normal(size=(200000, 3)) / np.sqrt(Om); x = y / np.linalg.norm(y, axis=1)[:, None]; return float(np.mean(np.exp(-np.sum(lam * x ** 2, 1)) * np.sum(Om * x ** 2, 1) ** 1.5 / M))
    As = np.logspace(0, 2.3, 12); res["acc_bingham_a"] = As.tolist(); res["acc_bingham"] = [acc_bing([0, 0, a]) for a in As]; out["samplers"] = res; print("samplers done", flush=True); save()
save(); print("all done")
