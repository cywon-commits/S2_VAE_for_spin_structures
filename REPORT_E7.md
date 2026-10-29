# REPORT_E7 — (1) refitted prior and non-linear response, (2) wave-vector resolved channel response χ(q) (nematic, no retraining)

Request `REQUEST_E7_PRIOR_QRESP_v1.md`; continues `REPORT_E5.md`, `REPORT_E6.md`. Model A = v3h (T=0.95, 1.05; R=2,3,4), model B = v5a (T=1.05; R=2,3,4). Encoders and decoders untouched. Per site, J=1, aggregated posterior = validation configurations × draws as in E5/E6. Code: `qresp.py`, `ana_E7q.py`, `run_E7_q.py`, `make_E7q.py` (task 2); `e7_prior.py`, `run_E7_1.py`, `run_E7_C.py`, `make_E7_1.py` (task 1); builder `build_report_E7.py`; raw output `results/response/E7_*`; figures `figures/E7_*`.

## Verdict
1. **The prior hole is entirely in the global latent, in the main l=2 channel.** The aggregated posterior of z_g sits on a thin shell in one l=2 channel: ‖z_g‖ = 3.9–4.2 ± 0.1 in the main channel (components 31–35) at T=1.05 (N(0,I) in 5 dimensions has ‖z‖ ≈ 2.1, E‖z‖² = 5; the posterior has E‖z‖² = 15.5–17.3); for model A at T=0.95 the large-norm channel is the other one (components 21–25, E‖z‖² = 11.6–13.7, the second channel 5.0–7.2). Decoding z_g from the posterior with cell deviations from the conditional prior (i) gives S = 0.52–0.56 at T=1.05 and 0.65–0.66 at T=0.95, i.e. close to the data (0.558 / 0.666); decoding z_g from the prior with the posterior's cell deviations (ii) gives S = 0.10–0.17 (the old-prior value). Cell noise is not the problem (ε variances 0.9–1.05 except the main-channel components of model B, 0.4–0.85, which are projected onto the rotation/amplitude subspaces; fitted std factors of those subspaces 1.01 / 1.05). The cell-level noise is not the problem .
2. **A rotation-augmented 50-component Gaussian mixture on z_g repairs the *mean* and the *non-linear director/biaxial* response.** Ancestral samples: S = 0.534–0.560 for model B, 0.520–0.542 for model A at T=1.05 (data 0.558) and 0.650–0.656 at T=0.95 (data 0.666). Latent HMC with the refitted prior (model B, R=2, T=1.05) reproduces MC: S(D=0) = 0.560 vs 0.559; director pinning ⟨P₂(d·e)⟩ = 0.755 at D=0.005 (MC 0.760), 0.904 at 0.01 (0.911); easy-plane alignment −0.453 / −0.474 / −0.493 at D = −0.01 / −0.02 / −0.05 (MC −0.461 / −0.476 / −0.490); easy-plane biaxial ⟨B⟩ = 0.016 / 0.007 / −0.006 (MC 0.013 / 0.006 / −0.003) — in E6 the VAE gave −0.08…−0.11 (>10× MC). Criteria (i), (ii), (iv) are met.
3. **Not repaired: the amplitude (S) response in the pinned window.** The slope dS/dD over D = 0.01…0.05 is 4.97 (HMC) vs 0.65 (MC), and easy-plane 2.53 vs 0.38: criterion (iii) fails by a factor ≈ 7. The cause is the width of S: refit-prior S has sd 0.038–0.048 (data 0.022 at T=1.05, 0.014 at T=0.95), variance ratio 3.4–7.4; since χ_Q,lat = βN Var(S) the ancestral χ_Q is 2.7–3.9 vs MC 0.90 / 0.39. Using k=200 components narrows S (0.037) but not to the data width; even the ideal case (i) (true posterior z_g + conditional-prior cells) has sd 0.030–0.034 for model B (0.024–0.027 for model A at T=1.05, 0.022–0.023 at T=0.95 vs data 0.014), because independent cell noise adds amplitude variance that the posterior's cell deviations (correlated with z_g and across cells) do not have. So fixing the amplitude channel needs a cell-level correlation structure, not a better z_g density.
4. **χ(q): the cell channels are carried exactly as far as the models' latent structure allows (task 2).** Decoder parts equal the MC site values in all channels (director 0.265–0.277 vs 0.275). At q≠0:
   - *director m=±1* (Frank mode): model B follows the prediction χ_site + (χ_MC−χ_site)|F_b|² (excess ratio 0.75–0.92 on the shells 0<k≤R/2); model A has **no director response beyond the site value** (0.01–0.02): its l=2 cell deviations are tiny (τ₂ = 0.105), so the Frank mode lives in a cell channel it does not use;
   - *amplitude m=0*: model B follows the prediction (1.10–1.27), model A carries only 0.24–0.41 of it at q≠0 while being 1.6–1.8× (T=1.05) / 2.7–3.0× (T=0.95) above MC at q=0 (global-latent noise);
   - *biaxial m=±2*: both are **far below** the prediction at q≠0 (A 0.06–0.08, B 0.12–0.35 of the predicted excess, B rising with R) and **above** MC at q=0 (A 1.21–1.45×, B 1.17–1.29×). So the biaxial cell channel is almost not transmitted in either model; model B (biaxial deviation tied to z_g) transmits *more* of it than model A, not less — H2 (A > B) is not supported;
   - above the cell wave number 2π/b all three channels fall to the site value (VAE/site 0.90–1.08).

## Task 1-a — diagnostics (aggregated posterior vs prior)
| model | T | R | E‖z‖² ch.1 / ch.2 (N(0,I): 5) | ‖z‖ main channel (mean ± sd) | z_g comp. variances (10 active; N(0,I): 1) | cell noise ε variances (active; N(0,I): 1) |
|---|---|---|---|---|---|---|
| B | 1.05 | 2 | 2.26 / 16.43 | 4.05 ± 0.11 | 0.46/0.45/0.45/0.45/0.45/1.56/0.93/3.57/2.47/1.72 | 1.03/1.05/1.05/1.01/1.04/0.63/0.42/0.57/0.85/0.63 |
| B | 1.05 | 4 | 0.05 / 15.49 | 3.93 ± 0.13 | 0.01/0.01/0.01/0.01/0.01/1.45/0.86/3.39/2.37/1.60 | 1.02/1.02/1.01/1.02/1.01/0.62/0.41/0.58/0.80/0.61 |
| B | 1.05 | 3 | 0.18 / 15.89 | 3.98 ± 0.12 | 0.04/0.04/0.04/0.04/0.04/1.52/0.90/3.47/2.39/1.63 | 1.03/1.04/1.03/1.04/1.03/0.63/0.42/0.60/0.84/0.62 |
| A | 1.05 | 2 | 4.99 / 17.31 | 4.16 ± 0.09 | 0.97/1.07/1.04/0.95/0.96/1.64/0.97/3.77/2.59/1.79 | 0.99/0.99/1.01/0.98/1.00/0.95/0.95/1.00/0.98/1.00 |
| A | 1.05 | 4 | 4.06 / 15.56 | 3.94 ± 0.10 | 0.81/0.82/0.77/0.80/0.86/1.48/0.87/3.39/2.33/1.62 | 1.00/1.00/1.00/1.00/1.00/0.98/0.93/1.00/1.00/0.98 |
| A | 1.05 | 3 | 4.63 / 15.80 | 3.97 ± 0.09 | 0.88/0.97/0.93/0.95/0.90/1.50/0.88/3.44/2.37/1.64 | 0.99/1.00/1.02/1.00/1.00/0.94/0.90/1.00/0.98/0.96 |
| A | 0.95 | 2 | 13.69 / 4.97 | 2.22 ± 0.21 | 0.94/0.29/1.34/2.51/0.49/0.43/0.22/0.57/0.89/0.23 | 0.96/0.94/0.95/1.03/1.04/0.99/1.00/0.99/1.03/1.00 |
| A | 0.95 | 4 | 11.61 / 7.18 | 2.67 ± 0.21 | 0.80/0.25/1.14/2.12/0.41/0.60/0.26/0.80/1.27/0.31 | 1.00/0.94/0.97/1.03/1.00/0.99/0.98/0.99/0.99/1.00 |
| A | 0.95 | 3 | 12.28 / 6.32 | 2.50 ± 0.22 | 0.86/0.26/1.20/2.24/0.44/0.53/0.24/0.70/1.13/0.28 | 0.97/0.92/0.98/1.05/1.02/0.99/0.98/0.99/1.01/1.00 |

Columns: "ch.1/ch.2" are the two active l=2 channels (components 21–25 and 31–35; the second is the main one, `act`=31). Prior: z_g ~ N(0,I) (E‖z‖²=5, variances 1); cell noise ε: N(0,I). Model B at R=3, 4 has a collapsed first channel (variance 0.01–0.04) – the posterior there lives only in the main channel.

S of ancestral samples (sample frame, per z; mean ± sd over samples; data = validation snapshots):

| model | T | R | S data | S post. | (i) z_g post, cells prior | (ii) z_g prior, cells post | old prior | refit prior (GMM k) | var ratio refit/data | var ratio (i)/data |
|---|---|---|---|---|---|---|---|---|---|---|
| B | 1.05 | 2 | 0.558±0.022 | 0.553±0.022 | 0.563±0.034 | 0.129±0.092 | 0.127±0.095 | 0.560±0.048 (k=50) | 4.7 | 2.3 |
| B | 1.05 | 4 | 0.558±0.022 | 0.535±0.022 | 0.535±0.030 | 0.168±0.093 | 0.169±0.095 | 0.534±0.041 (k=50) | 3.4 | 1.8 |
| B | 1.05 | 3 | 0.558±0.022 | 0.546±0.022 | 0.550±0.031 | 0.153±0.097 | 0.148±0.093 | 0.548±0.044 (k=50) | 4.0 | 2.0 |
| A | 1.05 | 2 | 0.558±0.022 | 0.555±0.030 | 0.545±0.024 | 0.104±0.084 | 0.104±0.085 | 0.542±0.043 (k=50) | 3.8 | 1.2 |
| A | 1.05 | 4 | 0.558±0.022 | 0.555±0.029 | 0.529±0.027 | 0.123±0.093 | 0.126±0.090 | 0.525±0.044 (k=50) | 4.0 | 1.4 |
| A | 1.05 | 3 | 0.558±0.022 | 0.555±0.030 | 0.522±0.026 | 0.117±0.090 | 0.116±0.087 | 0.520±0.045 (k=50) | 4.1 | 1.4 |
| A | 0.95 | 2 | 0.666±0.014 | 0.663±0.023 | 0.659±0.022 | 0.134±0.107 | 0.136±0.108 | 0.656±0.038 (k=50) | 7.0 | 2.4 |
| A | 0.95 | 4 | 0.666±0.014 | 0.663±0.024 | 0.653±0.023 | 0.141±0.110 | 0.143±0.109 | 0.650±0.039 (k=50) | 7.3 | 2.6 |
| A | 0.95 | 3 | 0.666±0.014 | 0.663±0.024 | 0.654±0.023 | 0.139±0.109 | 0.138±0.105 | 0.653±0.039 (k=50) | 7.4 | 2.6 |

(i)/(ii) are the decompositions requested in 1-a; the data S sd is 0.022 (var 4.9×10⁻⁴).

## Task 1-b — refit and linear check
Mixture: z_g posterior samples (active 10 components, 800 configs × 8 draws) × 16 random SO(3) rotations (the same D² Wigner matrix on both l=2 channels), full-covariance Gaussian mixtures, k ∈ {50, 100, 200} chosen by BIC (the request asked 20–50; BIC picked the smallest k=50 in every case, which is the lower BIC but not the best S width — see below), reg_covar 10⁻⁴. Cells: conditional prior as before (z_c = z_g + τε; no cell-level refit needed). Figure `E7_fig_Sdist`.

| model | T | R | χ_Q (agg. post.) | χ_Q old prior | χ_Q refit prior | χ_B (agg. post.) | χ_B old prior | χ_B refit prior | MC χ_Q / χ_B |
|---|---|---|---|---|---|---|---|---|---|
| B | 1.05 | 2 | 0.90 | 13.6 | 3.94 | 0.272 | 4.83 | 0.264 | 0.90 / 0.212 |
| B | 1.05 | 4 | 0.94 | 15.9 | 2.89 | 0.257 | 4.71 | 0.220 | 0.90 / 0.212 |
| B | 1.05 | 3 | 0.93 | 15.2 | 3.34 | 0.256 | 4.99 | 0.233 | 0.90 / 0.212 |
| A | 1.05 | 2 | 1.58 | 12.6 | 3.22 | 0.264 | 4.02 | 0.281 | 0.90 / 0.212 |
| A | 1.05 | 4 | 1.54 | 15.3 | 3.37 | 0.264 | 4.75 | 0.302 | 0.90 / 0.212 |
| A | 1.05 | 3 | 1.58 | 13.2 | 3.47 | 0.274 | 4.49 | 0.309 | 0.90 / 0.212 |
| A | 0.95 | 2 | 1.10 | 18.3 | 2.72 | 0.182 | 6.13 | 0.193 | 0.39 / 0.129 |
| A | 0.95 | 4 | 1.12 | 23.3 | 2.81 | 0.180 | 6.39 | 0.199 | 0.39 / 0.129 |
| A | 0.95 | 3 | 1.14 | 22.1 | 2.88 | 0.183 | 6.43 | 0.200 | 0.39 / 0.129 |

The refit lowers the old prior's χ (χ_Q 12.6–23, χ_B 4.0–6.4) to χ_Q 2.7–3.9 and χ_B 0.19–0.31 (aggregated posterior: χ_Q 0.9–1.6, χ_B 0.18–0.27; MC 0.90/0.39 and 0.212/0.129). The biaxial value is therefore repaired (model B 0.220–0.264 vs posterior 0.256–0.272; model A 0.28–0.31 vs 0.26–0.27 at T=1.05, 0.19–0.20 vs 0.18 at T=0.95), the amplitude value remains 2–4× too large (Verdict 3). Criterion "S mean within 0.02 of data": met for B R=2 (0.002), B R=3 (0.010), A T=0.95 (0.010–0.016), A T=1.05 R=2 (0.016); **not met** for B R=4 (0.024) and A T=1.05 R=3, 4 (0.038, 0.033; for model A the conditional-prior cells alone, case (i), already lower S by 0.01–0.036). "Variance ratio 0.7–1.4": **not met** (3.4–7.4).

## Task 1-c — non-linear response with the refitted prior (T=1.05)
HMC in the non-centred coordinates (z_g from the mixture, ε~N(0,I)); 12 chains, 100+150 steps (R=4: 8 chains, 40+60 steps, reduced D grid); MC = E6 task C.


**v3h_T1.05o_R2_b1** (T=1.05, R=2)

| D | S: MC / old / refit | ⟨P₂(d·e)⟩: MC / old / refit | ⟨B⟩: MC / old / refit | acc. | τ_int | chain-sd S/A/B (refit) |
|---|---|---|---|---|---|---|
| -0.050 | 0.579 / 0.637 / 0.677 | -0.490 / -0.491 / -0.492 | -0.003 / -0.109 / -0.006 | 0.63 | 1.9 | 0.014/0.003/0.002 |
| -0.020 | 0.568 / 0.453 / 0.584 | -0.476 / -0.461 / -0.474 | 0.006 / -0.080 / 0.006 | 0.62 | 2.4 | 0.013/0.009/0.004 |
| -0.010 | 0.564 / 0.172 / 0.564 | -0.461 / -0.280 / -0.444 | 0.013 / -0.024 / 0.019 | 0.70 | 3.4 | 0.009/0.013/0.005 |
| -0.005 | 0.562 / 0.110 / 0.549 | -0.412 / -0.175 / -0.390 | 0.033 / 0.016 / 0.040 | 0.66 | 3.1 | 0.013/0.028/0.010 |
| -0.002 | 0.561 / 0.106 / 0.541 | -0.329 / -0.087 / -0.263 | 0.064 / 0.023 / 0.085 | 0.73 | 3.5 | 0.007/0.159/0.055 |
| +0.000 | 0.559 / 0.100 / 0.540 | -0.047 / -0.014 / 0.037 | 0.169 / 0.028 / 0.193 | 0.80 | 2.2 | 0.005/0.261/0.093 |
| +0.002 | 0.561 / 0.111 / 0.545 | 0.271 / 0.079 / 0.542 | 0.289 / 0.047 / 0.379 | 0.43 | 5.0 | 0.008/0.214/0.079 |
| +0.005 | 0.563 / 0.133 / 0.560 | 0.760 / 0.234 / 0.763 | 0.473 / 0.080 / 0.472 | 0.77 | 2.3 | 0.005/0.062/0.021 |
| +0.010 | 0.568 / 0.401 / 0.576 | 0.911 / 0.707 / 0.891 | 0.535 / 0.358 / 0.534 | 0.82 | 1.9 | 0.010/0.016/0.010 |
| +0.020 | 0.576 / 0.730 / 0.633 | 0.945 / 0.957 / 0.942 | 0.555 / 0.710 / 0.608 | 0.74 | 3.0 | 0.010/0.007/0.010 |
| +0.050 | 0.594 / 0.846 / 0.787 | 0.980 / 0.986 / 0.987 | 0.586 / 0.838 / 0.783 | 0.64 | 2.2 | 0.005/0.005/0.007 |

**v5a_T1.05o_R2_b1** (T=1.05, R=2)

| D | S: MC / old / refit | ⟨P₂(d·e)⟩: MC / old / refit | ⟨B⟩: MC / old / refit | acc. | τ_int | chain-sd S/A/B (refit) |
|---|---|---|---|---|---|---|
| -0.050 | 0.579 / 0.664 / 0.676 | -0.490 / -0.493 / -0.493 | -0.003 / -0.096 / -0.006 | 0.70 | 2.3 | 0.010/0.002/0.001 |
| -0.020 | 0.568 / 0.425 / 0.601 | -0.476 / -0.457 / -0.474 | 0.006 / -0.081 / 0.007 | 0.70 | 1.7 | 0.009/0.010/0.005 |
| -0.010 | 0.564 / 0.215 / 0.575 | -0.461 / -0.329 / -0.453 | 0.013 / -0.019 / 0.016 | 0.74 | 1.8 | 0.010/0.010/0.004 |
| -0.005 | 0.562 / 0.156 / 0.570 | -0.412 / -0.187 / -0.370 | 0.033 / 0.009 / 0.049 | 0.71 | 1.7 | 0.010/0.043/0.018 |
| -0.002 | 0.561 / 0.126 / 0.562 | -0.329 / -0.104 / -0.239 | 0.064 / 0.030 / 0.096 | 0.69 | 3.5 | 0.008/0.120/0.045 |
| +0.000 | 0.559 / 0.126 / 0.560 | -0.047 / -0.008 / -0.064 | 0.169 / 0.046 / 0.162 | 0.74 | 1.8 | 0.009/0.154/0.056 |
| +0.002 | 0.561 / 0.127 / 0.565 | 0.271 / 0.052 / 0.391 | 0.289 / 0.048 / 0.336 | 0.60 | 2.3 | 0.007/0.194/0.074 |
| +0.005 | 0.563 / 0.175 / 0.583 | 0.760 / 0.297 / 0.755 | 0.473 / 0.118 / 0.490 | 0.69 | 3.1 | 0.010/0.037/0.011 |
| +0.010 | 0.568 / 0.432 / 0.603 | 0.911 / 0.756 / 0.904 | 0.535 / 0.395 / 0.565 | 0.76 | 2.3 | 0.011/0.023/0.011 |
| +0.020 | 0.576 / 0.743 / 0.657 | 0.945 / 0.961 / 0.956 | 0.555 / 0.723 / 0.638 | 0.63 | 2.1 | 0.011/0.007/0.012 |
| +0.050 | 0.594 / 0.845 / 0.802 | 0.980 / 0.985 / 0.984 | 0.586 / 0.837 / 0.792 | 0.74 | 1.5 | 0.014/0.005/0.013 |

**v5a_T1.05o_R2_b1 [GMM k=200]** (T=1.05, R=2)

| D | S: MC / old / refit | ⟨P₂(d·e)⟩: MC / old / refit | ⟨B⟩: MC / old / refit | acc. | τ_int | chain-sd S/A/B (refit) |
|---|---|---|---|---|---|---|
| -0.050 | 0.579 / 0.664 / 0.635 | -0.490 / -0.493 / -0.491 | -0.003 / -0.096 / -0.004 | 0.73 | 2.4 | 0.008/0.003/0.003 |
| +0.010 | 0.568 / 0.432 / 0.588 | 0.911 / 0.756 / 0.894 | 0.535 / 0.395 / 0.546 | 0.81 | 2.1 | 0.007/0.027/0.011 |
| +0.050 | 0.594 / 0.845 / 0.756 | 0.980 / 0.985 / 0.982 | 0.586 / 0.837 / 0.747 | 0.69 | 3.5 | 0.012/0.004/0.012 |

**v5a_T1.05o_R4_b1** (T=1.05, R=4)

| D | S: MC / old / refit | ⟨P₂(d·e)⟩: MC / old / refit | ⟨B⟩: MC / old / refit | acc. | τ_int | chain-sd S/A/B (refit) |
|---|---|---|---|---|---|---|
| -0.010 | 0.564 / 0.230 / 0.549 | -0.461 / -0.404 / -0.430 | 0.013 / -0.023 / 0.024 | 0.75 | 2.1 | 0.016/0.027/0.008 |
| -0.005 | 0.562 / – / 0.543 | -0.412 / – / -0.361 | 0.033 / – / 0.048 | 0.74 | 4.6 | 0.017/0.101/0.035 |
| +0.000 | 0.559 / – / 0.533 | -0.047 / – / 0.242 | 0.169 / – / 0.263 | 0.71 | 1.3 | 0.012/0.393/0.137 |
| +0.005 | 0.563 / – / 0.545 | 0.760 / – / 0.780 | 0.473 / – / 0.465 | 0.73 | 1.0 | 0.009/0.096/0.034 |
| +0.010 | 0.568 / 0.411 / 0.558 | 0.911 / 0.734 / 0.906 | 0.535 / 0.365 / 0.524 | 0.66 | 1.3 | 0.013/0.031/0.015 |
| +0.050 | 0.594 / 0.821 / 0.767 | 0.980 / 0.987 / 0.985 | 0.586 / 0.815 / 0.758 | 0.52 | 4.3 | 0.024/0.007/0.020 |

Criteria (T=1.05; slopes dS/dD from D = 0.01 to 0.05, MC 0.65; easy plane dS/d|D| from |D| = 0.01 to 0.05, MC 0.38):
- (i) S(D=0) within 0.02 of data (0.558): B R=2 0.560 ✓; A R=2 0.540 (0.018 ✓, its ancestral S is also low); B R=4 0.533 (0.025 ✗ — its own posterior S is 0.535).
- (ii) pinning at D=0.005, ⟨P₂(d·e)⟩ ≥ 0.7: B R=2 0.755, B R=4 0.780, A R=2 0.763 (MC 0.760) ✓.
- (iii) slope in the pinned window within 20 % of MC: **✗** — easy axis 4.97 (B R=2), 5.2 (B R=4), 5.3 (A R=2) vs 0.65 (k=200 mixture, B R=2: 4.2); easy plane 2.53 (B R=2) vs 0.38. The VAE is right for D ≤ 0.01 (S within 0.01–0.02 of MC, e.g. B R=2 D=0.01: 0.603 vs 0.568, D=−0.01: 0.575 vs 0.564) and departs at |D| ≥ 0.02 (0.60–0.80 vs 0.57–0.59).
- (iv) easy-plane ⟨B⟩ at the MC size: ✓ for |D| ≥ 0.01 (|⟨B⟩_VAE − ⟨B⟩_MC| ≤ 0.011, B R=2 0.016 / 0.007 / −0.006 vs 0.013 / 0.006 / −0.003; E6: −0.08…−0.11). At |D| = 0.002–0.005 the VAE is 0.016–0.03 above MC (director pinning is slightly softer: ⟨P₂(d·e)⟩(−0.002) = −0.24 (B), −0.26 (A) vs −0.33 in MC; at D=+0.002 0.39–0.54 vs 0.27), consistent with its larger S sd and with chain-to-chain sd up to 0.2 (A) at D=0.
The remaining non-linear discrepancy is therefore confined to S at |D| ≥ 0.02, i.e. the amplitude channel, the same channel that overestimates χ_Q,lat (Verdict 3).

## Task 2 — wave-vector resolved response χ(q)
MC: `data/ll_L12_T{0.950,1.050}_o.npy` (2000 snapshots; frame rule of `fdt_biax.py`); reproduces χ_Q(0) = 0.390 / 0.903 and χ_B(0) = 0.129 / 0.212. VAE: χ_dec (q-independent) + χ_lat(q), χ_lat(q)=(β/N) Var_z|Σ_i⟨O_i⟩_z e^{iq·r_i}|² from the cell-constant decoder means expanded onto the 12³ lattice (aggregated posterior, validation × 4 draws), shells in |k|² (q = 2π k/12). Prediction: χ_site + (χ_MC(k) − χ_site)|F_b(k)|² per k vector, then shell-averaged. Figures: `figures/E7_fig_chiq_T1.05.*` (model B and A, three channels), `E7_fig_chiq_T0.95.*` (model A). "Excess ratio" = (χ_VAE − χ_site)/(χ_pred − χ_site) averaged over the shells with 0 < k ≤ R/2 (low) and R/2 < k < R (mid; k=R is the cell wave number 2π/b, where F_b = 0).

| model | T | R | channel | χ_dec (MC site) | q=0: VAE / MC | excess ratio (VAE−site)/(pred−site), 0<k≤R/2 | same, R/2<k<R | VAE/site for k>R+1 |
|---|---|---|---|---|---|---|---|---|
| A | 1.05 | 2 | amplitude m=0 | 0.135 (0.135) | 1.441 / 0.903 | 0.37 | 0.49 | 1.01 |
| A | 1.05 | 2 | director m=±1 | 0.275 (0.275) | – | 0.01 | 0.03 | 1.00 |
| A | 1.05 | 2 | biaxial m=±2 | 0.072 (0.071) | 0.261 / 0.212 | 0.07 | 0.17 | 1.02 |
| A | 1.05 | 3 | amplitude m=0 | 0.134 (0.135) | 1.567 / 0.903 | 0.41 | 0.84 | 1.00 |
| A | 1.05 | 3 | director m=±1 | 0.274 (0.275) | – | 0.02 | 0.08 | 1.00 |
| A | 1.05 | 3 | biaxial m=±2 | 0.072 (0.071) | 0.266 / 0.212 | 0.08 | 0.39 | 1.02 |
| A | 1.05 | 4 | amplitude m=0 | 0.134 (0.135) | 1.598 / 0.903 | 0.29 | 0.99 | 1.00 |
| A | 1.05 | 4 | director m=±1 | 0.273 (0.275) | – | 0.02 | 0.06 | 1.00 |
| A | 1.05 | 4 | biaxial m=±2 | 0.073 (0.071) | 0.256 / 0.212 | 0.07 | 0.85 | 1.03 |
| A | 0.95 | 2 | amplitude m=0 | 0.101 (0.098) | 1.175 / 0.390 | 0.30 | 0.45 | 1.02 |
| A | 0.95 | 2 | director m=±1 | 0.277 (0.278) | – | 0.01 | 0.02 | 1.00 |
| A | 0.95 | 2 | biaxial m=±2 | 0.049 (0.048) | 0.186 / 0.129 | 0.07 | 0.25 | 1.02 |
| A | 0.95 | 3 | amplitude m=0 | 0.100 (0.098) | 1.049 / 0.390 | 0.32 | 0.73 | 1.02 |
| A | 0.95 | 3 | director m=±1 | 0.276 (0.278) | – | 0.02 | 0.03 | 0.99 |
| A | 0.95 | 3 | biaxial m=±2 | 0.049 (0.048) | 0.182 / 0.129 | 0.06 | 0.37 | 1.02 |
| A | 0.95 | 4 | amplitude m=0 | 0.100 (0.098) | 1.128 / 0.390 | 0.24 | 1.03 | 1.02 |
| A | 0.95 | 4 | director m=±1 | 0.276 (0.278) | – | 0.02 | 0.00 | 0.99 |
| A | 0.95 | 4 | biaxial m=±2 | 0.049 (0.048) | 0.178 / 0.129 | 0.07 | 0.85 | 1.02 |
| B | 1.05 | 2 | amplitude m=0 | 0.135 (0.135) | 0.891 / 0.903 | 1.10 | 1.34 | 1.01 |
| B | 1.05 | 2 | director m=±1 | 0.265 (0.275) | – | 0.92 | 0.88 | 0.97 |
| B | 1.05 | 2 | biaxial m=±2 | 0.073 (0.071) | 0.274 / 0.212 | 0.12 | 0.32 | 1.02 |
| B | 1.05 | 3 | amplitude m=0 | 0.132 (0.135) | 0.936 / 0.903 | 1.13 | 2.15 | 0.99 |
| B | 1.05 | 3 | director m=±1 | 0.253 (0.275) | – | 0.75 | 2.00 | 0.93 |
| B | 1.05 | 3 | biaxial m=±2 | 0.074 (0.071) | 0.247 / 0.212 | 0.22 | 1.07 | 1.05 |
| B | 1.05 | 4 | amplitude m=0 | 0.128 (0.135) | 0.956 / 0.903 | 1.27 | 2.76 | 0.97 |
| B | 1.05 | 4 | director m=±1 | 0.240 (0.275) | – | 0.84 | 1.24 | 0.90 |
| B | 1.05 | 4 | biaxial m=±2 | 0.077 (0.071) | 0.260 / 0.212 | 0.35 | 2.98 | 1.08 |

Reading: (a) the decoder part reproduces the MC single-site value in every channel (3 % or better; model B R=3, 4 director 0.253 / 0.240 vs 0.275 is 8–13 % low). (b) *Director channel* (G ≈ 4): model B is on the prediction (excess ratio 0.75–0.92; first shell k²=1: 1.82 vs 1.95 for R=2, 2.41 vs 3.10 for R=3, 2.87 vs 3.63 for R=4, i.e. 7 %, 22 %, 21 % low); model A has none (≤0.02) — invisible at q=0. (c) *Amplitude*: model B 1.10–1.27 (10–27 % above the prediction), model A 0.24–0.41 at q≠0 vs 1.6–3.0× at q=0: A's amplitude over-estimate of E5 is a q=0 (global latent) excess, while its cell channel under-transmits. (d) *Biaxial*: low-k excess ratio 0.06–0.08 (A) and 0.12–0.35 (B) with q=0 over-estimates 1.2–1.45×. Neither over- nor under-transmission is the single story: the global latent adds noise at q=0 (over) while the cell channel is absent (under). (e) Beyond 2π/b all channels collapse onto the site value within 10 %, as predicted by the shape factor. The "mid" column (R/2<k<R) is ill-conditioned (χ_pred − χ_site → 0 as k → R, so the ratio is noisy: 0.0–3.0); it is listed for completeness only.

## Criteria (request §2-c)
| item | result |
|---|---|
| 1 director channel within 10 % of prediction below 2π/b | model B: 7 % (R=2) / 22 % (R=3) / 21 % (R=4) low at the first shell, excess ratio 0.75–0.92 averaged over k ≤ R/2: **within 10 % only for R=2**; model A: **no** (0.01–0.02) |
| 2 amplitude: B on the prediction, A too large | B ✓ (1.10–1.27, 10–27 % above); A: too large at q=0 (1.6–3.0×) but **below** the prediction at q≠0 (0.24–0.41) |
| 3 biaxial: over (noise) or under (not transmitted); A vs B (H2) | both: over at q=0 (1.2–1.45×), under at q≠0 (0.06–0.35 of the excess); B ≥ A ⇒ H2 (A > B) not supported |
| 4 above 2π/b χ_VAE ≈ χ_site | ✓ (0.90–1.08) |

## Caveats
(i) The GMM is fitted per checkpoint to 800 independent configurations (× 8 draws × 16 rotations): the augmented data are not independent samples, so BIC is over-optimistic about large k and the chosen k=50 is the lower bound of the requested 20–50 range (all four model/T combinations). k=200 narrows S only modestly (0.037 vs 0.048). The VampPrior alternative was not tried. (ii) The cell-level prior was **not** refitted (variances ≈ 1); a zero-sum constraint on ε reduces the ideal-case S sd only from 0.034 to 0.028 (model B R=2); the remaining excess is correlation between z_g and the cells in the posterior that the factorised conditional prior cannot represent. (iii) R=4 HMC uses 8 chains, 40+60 steps; R=3 and T=0.95 HMC were not run. (iv) HMC k=200 sensitivity run: see table below the HMC results if present. (v) q-resolved data use the director d(z) defined from the decoded second moments; for the MC the frame comes from the realised Q — the same rule, but the VAE's d is an average over its cells. (vi) The q=0 shell of the director channel is excluded (frame definition); biaxial/amplitude q=0 use the global-mean-subtracted variance as in E6.
