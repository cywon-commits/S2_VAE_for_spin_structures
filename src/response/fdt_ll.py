"""Lebwohl-Lasher (J=1, z=6) H=0 data: response to (i) a Zeeman l=1 field, (ii) a quadrupolar field -eps sum P2(n.e) with e along the director.
MC: fluctuation-dissipation  chi = beta N Var.   Frozen-Weiss (exact single-site conditional in the local field, neighbours fixed): chi_W = beta E Var(.|rest)."""
import json, numpy as np
K = 3000; g = np.arange(K) + 0.5; th = np.arccos(1 - 2 * g / K); ph = np.pi * (1 + 5 ** 0.5) * g
G = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)          # Fibonacci grid
out = {}
for T in (0.80, 0.95, 1.05, 1.10):
    S = np.load(f"/home/claude/paper/repo/data/ll_L12_T{T:.3f}_o.npy", mmap_mode="r"); b = 1 / T
    snaps = np.asarray(S[200::4], float); N = snaps[0].size // 3                        # 450 snapshots after burn-in
    M1, Sp, Wl1, Wq = [], [], [], []
    for k, n in enumerate(snaps):
        X = n.reshape(-1, 3); Q = 1.5 * (X.T @ X) / N - 0.5 * np.eye(3); w, V = np.linalg.eigh(Q); d = V[:, 2]
        M = X.sum(0) @ V[:, ::-1]; M1.append(np.outer(M, M) / N)                       # body frame: director first
        Sp.append(np.mean(1.5 * (X @ d) ** 2 - 0.5))
        if k % 15 == 0:                                                                 # frozen-Weiss on 30 snapshots
            A = sum(np.roll(n, s, axis=ax) for ax in range(3) for s in (1, -1)).reshape(-1, 3)   # neighbour sum not usable for P2: build tensor
            Tn = sum(np.einsum('...i,...j->...ij', np.roll(n, s, axis=ax), np.roll(n, s, axis=ax)) for ax in range(3) for s in (1, -1)).reshape(-1, 3, 3)
            idx = np.random.default_rng(k).choice(N, 400, replace=False)
            E = 1.5 * b * np.einsum('ki,sij,kj->sk', G, Tn[idx], G); E -= E.max(1, keepdims=True); P = np.exp(E); P /= P.sum(1, keepdims=True)
            Gb = G @ V[:, ::-1]; nn = np.einsum('sk,ka,kb->sab', P, Gb, Gb); Wl1.append(b * nn.mean(0))
            p2 = 1.5 * (G @ d) ** 2 - 0.5; m1 = P @ p2; m2 = P @ p2 ** 2; Wq.append(b * np.mean(m2 - m1 ** 2))
    M1 = np.mean(M1, 0) / T; Sp = np.array(Sp)
    # block error for N Var(S)
    blk = Sp[: len(Sp) // 10 * 10].reshape(10, -1); vq = b * N * np.var(Sp); vq_se = b * N * np.std([np.var(x) for x in blk], ddof=1) / np.sqrt(10)
    Wl1 = np.mean(Wl1, 0)
    r = dict(S=float(Sp.mean()), zeeman_fdt_par=float(M1[0, 0]), zeeman_fdt_perp=float((M1[1, 1] + M1[2, 2]) / 2),
             zeeman_curie_par=float(b * (2 * Sp.mean() + 1) / 3), zeeman_curie_perp=float(b * (1 - Sp.mean()) / 3),
             zeeman_weiss_par=float(Wl1[0, 0]), zeeman_weiss_perp=float((Wl1[1, 1] + Wl1[2, 2]) / 2),
             quad_fdt=float(vq), quad_fdt_se=float(vq_se), quad_weiss=float(np.mean(Wq)))
    out[T] = r; print(T, {k: round(v, 4) for k, v in r.items()}, flush=True)
json.dump(out, open("fdt_ll.json", "w"), indent=1)
