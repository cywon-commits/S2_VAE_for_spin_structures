# REPORT_E3_v2 — site-resolved decoder and gain decomposition (kagome)

Follows `REPORT_E3.md`; request `REQUEST_E3_KAGOME_v2.md`. Same data, equivariance and seed-level split (train seeds 1–32, validation 33–40). "±" = standard error over the 8 validation seeds (C: seed-clustered). One training run per cell.

**Scope actually done:** A for R=1 and R=4 at T=0.0005 and 0.002; C for R=4 (v1 and v2 models); D at R=4. **Not done:** B (size-separated decoder with l=3 magnitude separation and subspace posterior covariance), the amortization-gap comparison that goes with it, A for T=0.001/0.004, and A for R=12 (a retrained R=12 would be identical in structure to the v1 R=12 model, since a cell is then a single unit cell and the v1 decoder already has per-site λ; the v1 R=12 numbers in `REPORT_E3.md` apply).

## A. Site-resolved decoder
Implementation (`src/kag/kmodel6.py`): the decoder produces K=12 cell-level multipole sets (l=1..8) from z_c and mixes them with a learned, rotation-invariant per-position embedding (position = sublattice × coordinates inside the cell), giving one λ_i per site; likelihood per site; encoder and up/down prior as before (R=1: plain N(0,I) prior). Equivariance error 1.2e-6. Training: encoder warm-started from the v1 up/down run (R=1: from the v1 R=1 model), decoder from scratch, 8000 steps. Note: this is a *low-rank* position dependence (rank ≤12 over 27 positions at R=4, over 432 at R=1), not a fully separate head per position; neighbouring-cell latents were not used.

| model | NLL/site | KL | net ELBO gain (nats/cfg) | NLL gain/site | -ln m3 | ln m6/ln m3 | ln m9/ln m3 | eps2 |
|---|---|---|---|---|---|---|---|---|
| **T=0.0005** | | | | | | | | |
| R=1 site-resolved (v2) | 0.119 | 57.6 | -17.0 ± 2.9 | -0.024 ± 0.007 | 0.081 | 1.72 | 3.30 | 0.114 |
| R=4 cell-const. (v1) | -0.488 | 221.5 | 81.0 ± 3.5 | 0.583 ± 0.014 | 0.068 | 1.62 | 2.89 | 0.084 |
| R=4 site-resolved (v2) | -0.557 | 234.8 | 97.7 ± 3.4 | 0.652 ± 0.015 | 0.068 | 1.52 | 2.67 | 0.088 |
| **T=0.002** | | | | | | | | |
| R=1 site-resolved (v2) | 1.266 | 34.6 | -4.0 ± 1.4 | -0.011 ± 0.003 | 0.264 | 1.96 | 3.40 | 0.231 |
| R=4 cell-const. (v1) | 0.506 | 202.1 | 156.8 ± 5.7 | 0.749 ± 0.023 | 0.152 | 1.92 | 3.26 | 0.167 |
| R=4 site-resolved (v2) | 0.427 | 178.9 | 214.2 ± 8.9 | 0.828 ± 0.026 | 0.144 | 1.90 | 3.19 | 0.162 |

Against the three predictions:
1. **R=4 harmonic ratios / ε² move toward the v1 R=12 values: not observed.** The site-resolved R=4 decoder has ε² 0.088/0.162 and ratios 1.52/2.67 (T=0.0005), 1.90/3.19 (T=0.002), essentially the v1 R=4 values (0.084; 1.62/2.89).
2. **R=4 NLL gain larger than v1: yes, modestly.** NLL gain 0.652 vs 0.583 nats/site (T=0.0005) and 0.828 vs 0.749 (T=0.002); net ELBO gain 97.7±3.4 vs 81.0±3.5 and 214.2±8.9 vs 156.8±5.7, with a slightly larger KL (235 vs 222, 179 vs ~170). Different training length/warm start between the two models was not controlled, so the size of the improvement is indicative only.
3. **R=1 improves: not observed.** The site-resolved R=1 model is slightly *worse* than the v1 R=1 model (net −17.0±2.9 and −4.0±1.4; NLL gain −0.024, −0.011).

Additional diagnostic (from `cellframe_k.py`, which measures the *data* in per-cell frames, R=4):

| T/J | data in own cell frames (R=4) | decoder-frames on data | shuffled control |
|---|---|---|---|
| 0.0005 | 1.49 / 2.51, ε² 0.083 | 1.46 / 2.35, ε² 0.086 | 1.75 / 3.08, ε² 0.103 |
| 0.002 | 2.01 / 3.31, ε² 0.151 | 2.00 / 3.29, ε² 0.158 | 2.05 / 3.36, ε² 0.196 |

The data's own harmonic ratios in 3×3-cell frames (1.49/2.51, ε² 0.083 at T=0.0005) already agree with the decoder's (1.52/2.67, 0.088), and the site-resolved decoder reproduces them. So at R=4 the low ratios are not shown to be a decoder-structure limit; they are what the data look like in frames defined from 27 spins. The high R=12 values (3.7/8.2) are then not directly comparable: a frame fitted per 3-spin cell absorbs fluctuations (cf. the shuffled controls, which already look "ordered" in own-frame statistics). I did not compute the data-own-frame statistics at R=12, so this reading of the R=12 numbers is a hypothesis, not a result.

## C. Gain decomposition regression
y = N·[NLL(R=1) − NLL(R)] per validation snapshot (N=432, reconstruction part only, KL excluded), regressed on N·g_ring (per-snapshot ring-oracle gain from in-sample per-sublattice L=8 fits pooled over the T's validation snapshots; ring correction as in `ring_oracle.py`) and n_def (3×3-block phase defects, rule of `kagome/block_defects.py`; mean n_def = 0.05 at T=0.0005, 1.36 at T=0.002 vs 0, ~1.1 quoted in the request). Standard errors clustered by validation seed.

| T | R-model | a (coef. of N·g_ring) | b (nats per defect) | c (nats) | R² |
|---|---|---|---|---|---|
| 0.0005 | v1 R=4 | 0.12 ± 0.06 | — | 236 ± 10 | 0.01 |
| 0.002 | v1 R=4 | -0.14 ± 0.06 | 14.5 ± 1.3 | 320 ± 9 | 0.29 |
| pooled | v1 R=4 | -0.04 ± 0.07 | 21.7 ± 2.3 | 277 ± 15 | 0.29 |
| 0.0005 | v2 site R=4 | 0.11 ± 0.06 | — | 267 ± 10 | 0.01 |
| 0.002 | v2 site R=4 | -0.12 ± 0.07 | 16.2 ± 1.2 | 349 ± 11 | 0.31 |
| pooled | v2 site R=4 | -0.04 ± 0.08 | 23.5 ± 2.2 | 308 ± 15 | 0.31 |

Prediction a≈1, b>0, small c: **b>0 holds** (≈15–16 nats per defect at T=0.002, 22–23 pooled), but **a≈1 does not** (a = 0.1 and −0.1 at the two temperatures, pooled −0.04±0.08), and **c is not small** (≈240–350 nats, i.e. most of y is a constant per snapshot). Both models behave alike. Likely readings (not tested): the ring-oracle per-snapshot variation is small compared with other snapshot-to-snapshot variation (R² ≈ 0.01 at T=0.0005), so a is poorly determined and the zero-mode channel cannot be isolated from the constant; or the VAE does not recover the loop-rotation channel through R=4 cells. The regression cannot distinguish these. Scatter plots: `results/kag/regress_*_R4.png`.

## D. Ablation by refit in the decoder frame
Per site, the frame is z′ = plane normal (min eigenvector of the decoded site Q tensor), x′ at the 3-fold phase of the decoded site distribution (posterior-mean decode); data spins are rotated into these frames and a multipole family of order L is refit by ML over 40 validation snapshots (pooled, in-sample).

| T/J | L=1 | L=2 | L=3 | L=4 | L=6 | L=8 |
|---|---|---|---|---|---|---|
| 0.0005 | 2.523 | 1.928 | 0.196 | 0.188 | -0.298 | -0.364 |
| 0.002 | 2.529 | 2.220 | 0.912 | 0.892 | 0.694 | 0.681 |

L=2→3 gives the large drop (T=0.0005: 1.93→0.196; T=0.002: 2.22→0.912) and the NLL then saturates monotonically (L=6→8: 0.07 and 0.01 nats), matching the prediction and removing the L=1/L=2 non-monotonicity of the unrefit truncation table in `REPORT_E3.md`. Caveat: a first attempt used the decoded site mean direction as z′, which is meaningless because decoded site distributions are three-cap mixtures (mean vector length ≈0.14); that result (NLL ≈2.0–2.5 for all L) was discarded and the file overwritten.

## Not done / next
B is not implemented, so the prediction that size separation plus subspace covariance lifts R=12 low-T net gain from ≈0 is untested. A possible follow-up is B on the R=4 site model, where the amortization-gap result of `REPORT_E3.md` (41 nats/config at T=0.0005) suggests encoder inefficiency is real.
