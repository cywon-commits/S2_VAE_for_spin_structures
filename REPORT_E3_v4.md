# REPORT_E3_v4 — gauge-free colour input (cap-direction vector) and the channel-locking hypothesis

Follows `REPORT_E3_v3.md`; request `REQUEST_E3_KAGOME_v4.md`. Data, split (validation = 8 seeds) and R=4 site-resolved model as in v2. "±" = seed s.e.; regressions are seed-clustered. One training run per cell.

## Verdict
With a gauge-free colour input the decoder **does use the colouring** (decoded site distributions become single caps and NLL drops by ≈1.1 nats/site), so the v3 failure was indeed an input-design problem. **But the key prediction of the channel-locking hypothesis is not confirmed:** the gain that the latent adds on top of the colouring is **not** explained by the per-snapshot ring-oracle gain (a = 0.16±0.04 at T=0.0005, 0.05±0.08 at T=0.002; predicted ≈1), and it is several times larger than the ring-oracle gain, with most of it in the intercept.

## Setup
- Input: per site, the triad direction closest to the spin, u_{c_i} (global_triad of the configuration itself; for time-reversed training copies recomputed from −S, so u transforms like the data). Entry into the decoder (`src/kag/kmodel6u.py`): (a) a tensor product of Y_1..3(u_i) with a linear projection of the cell latent (equivariant, can rotate/displace the cap), (b) a cap term s_l·Y_l(u_i) whose shape scalars depend on position and the l=0 latents, on top of the v2 low-rank pathway. Equivariance error 8.6e-9 (double precision). The mixing-weight-only entry of v3 is not used.
- Training: **from scratch** with the v2 recipe (encoder warm start from the v1 up/down run, decoder from scratch, 8000 steps, same lr and KL warm-up) for all of (ii), (iii), (iv). (iv): latent fixed to 0 (zmask=0, KL=0). (iii): u_c randomly permuted among the sites of each snapshot (fixed seed).
- Not done: (iii)/(iv)-type controls for R=1, R=12, T=0.001/0.004, task B (optional). Ring counting for the "5 nats × number of rings" estimate was not performed.

## Results
| T | model | NLL/site ± se | KL | decoded mean-vector length | decoded θ² (rad²) |
|---|---|---|---|---|---|
| 0.0005 | (i) v2 R=4, no colour | -0.557 | 234.8 | ≈0.14 (v2 report) | — |
| 0.0005 | (ii) R=4 + u_c (oracle) | -1.662 ± 0.080 | 201.1 ± 1.0 | 0.948 | 0.110 |
| 0.0005 | (iii) shuffled u_c | -0.530 ± 0.076 | 210.4 ± 1.9 | 0.079 | 2.873 |
| 0.0005 | (iv) no latent, u_c only | -0.230 ± 0.070 | 0.0 ± 0.0 | 0.938 | 0.131 |
| 0.002 | (i) v2 R=4, no colour | 0.427 | 178.9 | ≈0.14 (v2 report) | — |
| 0.002 | (ii) R=4 + u_c (oracle) | -0.641 ± 0.085 | 173.8 ± 1.6 | 0.899 | 0.219 |
| 0.002 | (iii) shuffled u_c | 0.429 ± 0.078 | 187.6 ± 1.8 | 0.060 | 2.854 |
| 0.002 | (iv) no latent, u_c only | 0.662 ± 0.097 | 0.0 ± 0.0 | 0.885 | 0.245 |

- Criterion 1 (single caps, mean-vector length ≥0.8): **met** for (ii) 0.948 / 0.899 and (iv) 0.938 / 0.885 (T=0.0005 / 0.002); not for the shuffled control (0.08 / 0.06).
- Criterion 4 ((iii) ≈ (i)): met at T=0.002 (NLL 0.429 vs 0.427, KL 188 vs 179); at T=0.0005 (iii) is 0.027 nats/site (≈12 nats per snapshot) worse than (i) (−0.530 vs −0.557) with 24 nats less KL, i.e. roughly equal ELBO to within the noise I can resolve here.
- Colour information: (ii) vs (i) NLL gain = 1.105 (T=0.0005) and 1.067 (T=0.002) nats/site, close to the upper bound ln 3 = 1.099 for a 3-colouring; this is conditioning information that is not charged in the KL, so it is not a latent gain. I did not compute the actual entropy of the colouring.
- Net ELBO value of the latent given colours, ELBO(ii) − ELBO(iv): **417.6 ± 6.4** (T=0.0005), **388.9 ± 4.0** (T=0.002) nats/snapshot. ELBO(ii) − ELBO(iii) (value of colours when the latent is present): 499 ± 1 and 476 ± 3.

## Regression (y = N·[NLL(iv) − NLL(ii)] on N·g_ring and n_def; N=432)
| T | y | mean y (nats) | a ± se | b ± se (nats/defect) | c ± se | R² |
|---|---|---|---|---|---|---|
| 0.0005 | (iv)-(ii) NLL [main] | 619 | 0.161 ± 0.043 | — | 597 ± 9 | 0.02 |
| 0.0005 | (iv)-(iii) shuffled control | 129 | 0.122 ± 0.045 | — | 113 ± 7 | 0.01 |
| 0.0005 | R=1 - (i) colour-free v2 | 282 | 0.108 ± 0.065 | — | 267 ± 10 | 0.01 |
| 0.002 | (iv)-(ii) NLL [main] | 563 | 0.053 ± 0.079 | 15.3 ± 1.2 | 536 ± 10 | 0.26 |
| 0.002 | (iv)-(iii) shuffled control | 101 | -0.036 ± 0.099 | 18.7 ± 1.4 | 79 ± 13 | 0.35 |
| 0.002 | R=1 - (i) colour-free v2 | 358 | -0.117 ± 0.071 | 16.2 ± 1.2 | 349 ± 11 | 0.31 |

Predictions:
- **a ≈ 1: not met.** Main target a = 0.16 ± 0.04 (T=0.0005; no defect term since n_def̄ = 0.05) and 0.05 ± 0.08 (T=0.002), versus 0.11 / −0.12 for the colour-free model (R=1 − (i)). The shuffled control is also near zero.
- **Intercept dominates:** mean y = 619 (T=0.0005) and 563 (T=0.002) nats per snapshot, of which c = 597 and 536. The mean ring-oracle quantity N·g_ring is only 133 and 112 nats (in-sample, KL not charged). So the latent's gain given colours is ≈4–5× the ring-oracle reconstruction gain and does not vary with it from snapshot to snapshot.
- b > 0 holds again at T=0.002 (15.3 ± 1.2 nats per defect; 18.7 ± 1.4 in the shuffled control, so the defect coefficient does not need colours).
- Net ELBO value (criterion 3): +418 and +389 nats per snapshot. Whether this is "within a factor 2 of the ring-rotation channel" cannot be said: I did not count rings, and since a ≈ 0 the gain is not attributable to ring rotations by this regression.

## Interpretation (hypotheses, not tested here)
- What the latent adds beyond the colouring is large (≈1.4 nats/site) but not ring-specific. Candidates: smooth long-wavelength deformations of the three-colour pattern inside a 3×3 cell (spin-wave-like modes, not only loop rotations), local triad/plane-orientation fluctuations, or cell-to-cell variation of the cap widths. A per-snapshot regressor for these (e.g. coherent in-cell rotation of u_c relative to the global triad) was not built.
- The ring-oracle g_ring is itself computed from the same colouring (loops are colour-defined), so the colouring in (ii) provides the information needed to define it; a ≈ 0 therefore says the *per-snapshot variation* of g_ring is not what the latent's extra gain tracks, not that ring information is absent from the latent.
- The premise that the colouring is the missing ingredient is supported (large NLL effect, single-cap decoding); the specific channel-locking prediction that the freed channel is the ring zero-mode channel is not.
