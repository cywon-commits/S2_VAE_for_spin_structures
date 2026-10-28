# REPORT_E6 — single-ion anisotropy as a two-channel probe (nematic, no retraining)

Request `REQUEST_E6_PERTURBATION_v1.md`; continues `REPORT_E5.md`. Checkpoints: v3h (T=0.95, 1.05; R=2,3,4) = "model A", v5a (T=1.05; R=2,3,4) = "model B". Per site, J=1. MC references: `fdt_biax.json` (chat-side, used as given): χ_Q = 0.39 / 0.90, χ_B = 0.129 / 0.212, χ_B,site = 0.048 / 0.071 (T=0.95 / 1.05). Code: `e6_response.py`, `run_E6_A.py`, `run_E6_B.py`, `ll_aniso_mc.py`, `ana_C6.py`, `make_E6.py`, `build_report_E6.py`; raw output `results/response/E6_*.json`, `C_aniso_T*.jsonl`; figures `figures/E6_fig1_channels`, `E6_fig2_nonlinear`.

## Verdict
1. **Implementation check (criterion 1): passed.** The m=0 operator reproduces the E5 numbers (model B χ_Q = 0.90/0.93/0.93 = 1.00–1.03 × FDT; model A 1.7× at T=1.05, 2.7–2.9× at T=0.95). The m=±2 decoder part χ_B,dec = 0.049 (T=0.95) / 0.072–0.076 (T=1.05) equals the MC site value 0.048 / 0.071 for every model, i.e. the independent-site part is exactly right (as for χ_Q,site ≈ χ_W in E5). Generated-batch checks agree with the analytic values (last column of the table).
2. **H1 (G sets the reliability): half supported.** Biaxial response is over-, not under-estimated: χ_B,VAE/χ_B,FDT = 1.38–1.42 (A, T=0.95), 1.24–1.27 (A, T=1.05), 1.21–1.33 (B, T=1.05). Compared with the amplitude channel the biaxial ratio is *smaller* for model A (1.24–1.42 vs 1.67–2.92) — as H1 predicts — but *larger* for model B (1.2–1.3 vs 1.00–1.03). So the prediction "biaxial under-estimated, ≲ χ_B,site + global share" fails: the VAE collective biaxial part (0.13–0.21) exceeds the MC collective part (0.141 at T=1.05; 0.081 at T=0.95) by 1.4–1.9×.
3. **H2 (response separates what ELBO cannot): not supported.** χ_B,VAE of model A (0.262/0.268/0.264) and model B (0.282/0.256/0.258) at T=1.05, R=2/3/4 are equal within the estimator's noise (±0.01 from `B_lat_se`, ±0.03 including the variation with R). Tying the cell biaxial deviation to the global latent in model B does **not** reduce the biaxial response (it is carried by the global latent). The amplitude channel does separate the two models (1.7 vs 1.0), but in the opposite sense of H2 — model B is the better one.
4. **Decomposition (A-3, criterion 5): the collective biaxial part is not from cell deviations.** Removing all cell deviations (global latent only) leaves χ_B,lat unchanged for model A (0.135 → 0.135 at T=0.95; 0.19 → 0.19 at T=1.05) and reduces it by 0.03–0.05 (15–27 %) for model B (0.18–0.21 → 0.13–0.18); the biaxial-deviation subspace contributes nothing in either, and the director-rotation subspace gives ≤ 0.06 (model B) / ≈0 (model A). So (a) the biaxial channel is carried by the *global* latent, and (b) it is not mainly director rotation: the "second-order director fluctuation" reading of the MC collective part does not apply to the model's excess. For the amplitude channel the same removal leaves χ_Q,lat of model A unchanged (1.38–1.47 → 1.38–1.47) and increases it for model B (0.77–0.80 → 0.99–1.72, i.e. in model B the cell rotation/amplitude deviations *reduce* the amplitude-response variance).
5. **Non-linear response (criterion 4): not reproduced.** The VAE ensembles (prior p_θ, which has the hole of E5: S(D=0)=0.10–0.14 vs 0.56–0.66 in MC) respond to D quantitatively differently from MC: S(D) rises steeply at |D| ≈ 0.01–0.02 (0.13 → 0.74 for model B at D=0.02) while MC S(D) stays within 0.56 → 0.59; the director alignment ⟨P₂(d·e)⟩ reaches 0.76 at D=0.005 in MC but only 0.2–0.3 in the VAE ensembles. Easy-plane ⟨B⟩ is **over-** rather than under-estimated: HMC gives −0.08 … −0.11 at D = −0.02 … −0.05 (all three R=2 models, T=1.05) against +0.006 … −0.003 in MC (Fig. 2c) — the VAE ensembles carry a strong biaxial response that the true model lacks at the same D (part of it is the larger director alignment/amplitude of the pinned VAE state, so this is not a clean single-channel statement). Details below.

## Setup (what was done)
- **Task A.** E5 machinery (analytic, aggregated posterior: validation configs × 8 draws; priors K=1000 reported separately). Operators: m=0 P₂(n·d); m=±2 (B, C) with B=(n·e1)²−(n·e2)², C=2(n·e1)(n·e2), frame per z as in the request (d = top eigenvector of the decoded Σ⟨nnᵀ⟩, e1 = lab-x projected ⟂ d, e2 = d×e1). χ_B = (β/N)·½[E_z Σ_i(Var B_i + Var C_i) + Var_z ΣB̄ + Var_z ΣC̄] (mean over z subtracted; the means are ≤10⁻³, so this equals the second moment). "Subspace" runs: z_g := cell mean of the sampled latent, cell deviation of the main l=2 channel projected onto the biaxial / rotation / rotation+amplitude subspace of z_g's 5-vector (projectors as in `mvae4.projectors`), all other deviations zeroed; "global only" = all deviations zeroed. For model A (MVAE3) the main channel is taken as the active l=2 channel with largest mean |z|.
- **Task C.** `ll_aniso_mc.py`: Bingham heat bath with A += (3/2)βD e eᵀ, e = ẑ, L=12, random start, 1000 equilibration + 3000 measurement sweeps (every 5th measured), 4 seeds, T=0.95, 1.05, D ∈ {0, ±0.002, ±0.005, ±0.01, ±0.02, ±0.05}. Observables per snapshot: S=⟨P₂(n·d)⟩, A=P₂(d·e), B = ⟨(n·e)²−(n·e')²⟩ with e'=d×e/|d×e|.
- **Task B.** `run_E6_B.py`: D enters as Δλ₂ₘ = βD(4π/5)Y₂ₘ(e), e=ẑ. (1) exact importance reweighting of K=4000 prior samples with random e per z; (2) latent HMC in the non-centred coordinates (z_g, ε) with z_c = z_g + τ_l ε (model A) or the rotation/amplitude-subspace map of MVAE8 (model B; projectors detached in the gradient — the Metropolis step uses the exact potential), 12 chains, 100 warm-up + 150 sampling steps, 12 leapfrog steps, step size adapted to 0.7 acceptance. For R=4 only D = −0.01, +0.01, +0.05 with 8 chains and 40+60 steps (cost); R=3 and T=0.95 R=4 were not run.

## Task A — channel-resolved χ (aggregated posterior)
| model | T | R | χ_Q dec | χ_Q lat | χ_Q VAE | χ_Q FDT | ratio Q | χ_B dec | χ_B lat | χ_B VAE | χ_B FDT (site) | ratio B | gen-check χ_B (60 z) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A (v3h) | 0.95 | 2 | 0.100 | 1.04 | 1.14 | 0.39 | 2.92 | 0.049 | 0.135 ± 0.003 | 0.184 | 0.129 (0.048) | 1.42 | 0.176 |
| A (v3h) | 0.95 | 3 | 0.099 | 0.98 | 1.08 | 0.39 | 2.77 | 0.049 | 0.134 ± 0.003 | 0.183 | 0.129 (0.048) | 1.42 | 0.167 |
| A (v3h) | 0.95 | 4 | 0.099 | 0.95 | 1.05 | 0.39 | 2.68 | 0.049 | 0.129 ± 0.003 | 0.178 | 0.129 (0.048) | 1.38 | 0.164 |
| A (v3h) | 1.05 | 2 | 0.136 | 1.37 | 1.50 | 0.90 | 1.67 | 0.072 | 0.190 ± 0.005 | 0.262 | 0.212 (0.071) | 1.24 | 0.245 |
| A (v3h) | 1.05 | 3 | 0.134 | 1.40 | 1.54 | 0.90 | 1.71 | 0.072 | 0.196 ± 0.005 | 0.268 | 0.212 (0.071) | 1.27 | 0.271 |
| A (v3h) | 1.05 | 4 | 0.134 | 1.38 | 1.52 | 0.90 | 1.69 | 0.072 | 0.191 ± 0.005 | 0.264 | 0.212 (0.071) | 1.24 | 0.251 |
| B (v5a) | 1.05 | 2 | 0.136 | 0.77 | 0.90 | 0.90 | 1.00 | 0.073 | 0.209 ± 0.005 | 0.282 | 0.212 (0.071) | 1.33 | 0.277 |
| B (v5a) | 1.05 | 3 | 0.132 | 0.80 | 0.93 | 0.90 | 1.03 | 0.074 | 0.182 ± 0.004 | 0.256 | 0.212 (0.071) | 1.21 | 0.266 |
| B (v5a) | 1.05 | 4 | 0.128 | 0.80 | 0.93 | 0.90 | 1.03 | 0.076 | 0.182 ± 0.005 | 0.258 | 0.212 (0.071) | 1.22 | 0.294 |

### A-3 decomposition of the latent part χ_lat
| model | T | R | χ_B lat, full | global only (deviations removed) | + biaxial dev. | + rotation dev. | + rotation+amplitude dev. | χ_Q lat full / global only / +rot+amp |
|---|---|---|---|---|---|---|---|---|
| A | 0.95 | 2 | 0.135 | 0.135 | 0.135 | 0.135 | 0.135 | 1.04 / 1.04 / 1.04 |
| A | 0.95 | 3 | 0.134 | 0.135 | 0.135 | 0.135 | 0.135 | 0.98 / 0.99 / 0.99 |
| A | 0.95 | 4 | 0.129 | 0.129 | 0.129 | 0.129 | 0.129 | 0.95 / 0.96 / 0.95 |
| A | 1.05 | 2 | 0.190 | 0.190 | 0.191 | 0.190 | 0.189 | 1.37 / 1.38 / 1.37 |
| A | 1.05 | 3 | 0.196 | 0.199 | 0.200 | 0.199 | 0.195 | 1.40 / 1.47 / 1.41 |
| A | 1.05 | 4 | 0.191 | 0.195 | 0.196 | 0.195 | 0.190 | 1.38 / 1.47 / 1.40 |
| B | 1.05 | 2 | 0.209 | 0.180 | 0.180 | 0.211 | 0.207 | 0.77 / 0.99 / 0.77 |
| B | 1.05 | 3 | 0.182 | 0.141 | 0.141 | 0.191 | 0.179 | 0.80 / 1.40 / 0.80 |
| B | 1.05 | 4 | 0.182 | 0.133 | 0.133 | 0.196 | 0.176 | 0.80 / 1.72 / 0.81 |

(prior samples: χ_B,lat = 3.7–5.9, i.e. 20–40× too large because of the prior hole (E5); kept in `E6_A.json`.)

## Task C — MC with the anisotropy

**T = 0.95** (4 seeds, random start)

| D | S | ⟨P₂(d·e)⟩ | ⟨B⟩ |
|---|---|---|---|
| -0.050 | 0.674 ± 0.001 | -0.492 ± 0.001 | -0.001 ± 0.000 |
| -0.020 | 0.668 ± 0.000 | -0.479 ± 0.003 | 0.008 ± 0.001 |
| -0.010 | 0.667 ± 0.000 | -0.468 ± 0.006 | 0.013 ± 0.002 |
| -0.005 | 0.666 ± 0.000 | -0.412 ± 0.020 | 0.039 ± 0.009 |
| -0.002 | 0.666 ± 0.000 | -0.409 ± 0.020 | 0.040 ± 0.009 |
| +0.000 | 0.665 ± 0.000 | 0.191 ± 0.248 | 0.306 ± 0.110 |
| +0.002 | 0.665 ± 0.000 | 0.511 ± 0.149 | 0.448 ± 0.066 |
| +0.005 | 0.667 ± 0.000 | 0.836 ± 0.016 | 0.594 ± 0.007 |
| +0.010 | 0.668 ± 0.000 | 0.914 ± 0.008 | 0.630 ± 0.003 |
| +0.020 | 0.672 ± 0.000 | 0.962 ± 0.001 | 0.655 ± 0.001 |
| +0.050 | 0.682 ± 0.000 | 0.985 ± 0.001 | 0.675 ± 0.000 |

**T = 1.05** (4 seeds, random start)

| D | S | ⟨P₂(d·e)⟩ | ⟨B⟩ |
|---|---|---|---|
| -0.050 | 0.579 ± 0.001 | -0.490 ± 0.001 | -0.003 ± 0.001 |
| -0.020 | 0.568 ± 0.001 | -0.476 ± 0.002 | 0.006 ± 0.001 |
| -0.010 | 0.564 ± 0.001 | -0.461 ± 0.007 | 0.013 ± 0.002 |
| -0.005 | 0.562 ± 0.000 | -0.412 ± 0.014 | 0.033 ± 0.005 |
| -0.002 | 0.561 ± 0.001 | -0.329 ± 0.060 | 0.064 ± 0.023 |
| +0.000 | 0.559 ± 0.001 | -0.047 ± 0.196 | 0.169 ± 0.073 |
| +0.002 | 0.561 ± 0.001 | 0.271 ± 0.080 | 0.289 ± 0.030 |
| +0.005 | 0.563 ± 0.001 | 0.760 ± 0.019 | 0.473 ± 0.008 |
| +0.010 | 0.568 ± 0.000 | 0.911 ± 0.013 | 0.535 ± 0.005 |
| +0.020 | 0.576 ± 0.000 | 0.945 ± 0.005 | 0.555 ± 0.002 |
| +0.050 | 0.594 ± 0.001 | 0.980 ± 0.001 | 0.586 ± 0.001 |

Linear slopes read in the pinned window (|D| = 0.01–0.05):

| T | easy-axis slope dS/dD (D=0.01→0.05) | easy-plane slope dS/d|D| (0.01→0.05) = χ_Q/2 | 2× easy-plane | χ_Q FDT |
|---|---|---|---|---|
| 0.95 | 0.34 | 0.16 | 0.32 | 0.39 |
| 1.05 | 0.65 | 0.38 | 0.76 | 0.90 |

Easy-axis slope dS/dD = χ_Q and easy-plane slope dS/d|D| = χ_Q/2 are consistent with each other (ratio 1.7–2.3 for 2:1 expected) and with the FDT χ_Q (0.35 vs 0.39 at T=0.95; 0.65–0.75 vs 0.90 at T=1.05 — the slope in this window is below the T→0 FDT value because S(D) is concave, as noted in E5). Easy-plane ⟨B⟩ at |D| = 0.01–0.05 is small (+0.013 … −0.003 at T=1.05): it is dominated by the residual misalignment (A = −0.46…−0.49 instead of −½, e not exactly ⟂ d), so the predicted −¾|D|χ_B (−0.008 at |D|=0.05) cannot be isolated from the offset; the *change* of ⟨B⟩ between |D|=0.01 and 0.05 (−0.016 at T=1.05) is 2.5× the linear prediction (−0.006), and I read it as consistent with the linear value only up to the misalignment correction. (B at D=0 is 0.17–0.31 for random d because e is then not ⟂ d: the quantity is only meaningful for D<0.)

## Task B — VAE ensembles in the anisotropy field
| model | R | D | S (HMC) | ⟨P₂(d·e)⟩ | ⟨B⟩ | acc. | τ_int | chain-sd (S/A/B) | S (reweight, ESS) |
|---|---|---|---|---|---|---|---|---|---|
| v3h T=0.95 | 2 | -0.050 | 0.726 | -0.492 | -0.077 | 0.71 | 3.6 | 0.029/0.003/0.018 | 0.691 (1) |
| v3h T=0.95 | 2 | -0.020 | 0.543 | -0.473 | -0.087 | 0.64 | 1.9 | 0.051/0.012/0.031 | 0.615 (2) |
| v3h T=0.95 | 2 | -0.010 | 0.340 | -0.389 | -0.031 | 0.67 | 15.4 | 0.062/0.039/0.023 | 0.311 (92) |
| v3h T=0.95 | 2 | -0.005 | 0.189 | -0.284 | -0.006 | 0.97 | 5.0 | 0.090/0.136/0.035 | 0.175 (1950) |
| v3h T=0.95 | 2 | -0.002 | 0.169 | 0.004 | 0.046 | 0.82 | 10.3 | 0.121/0.307/0.096 | 0.143 (3674) |
| v3h T=0.95 | 2 | +0.000 | 0.140 | 0.023 | 0.049 | 0.68 | 1.9 | 0.013/0.064/0.009 | 0.137 (4000) |
| v3h T=0.95 | 2 | +0.002 | 0.144 | 0.083 | 0.066 | 0.76 | 1.8 | 0.022/0.075/0.013 | 0.145 (3499) |
| v3h T=0.95 | 2 | +0.005 | 0.246 | 0.374 | 0.190 | 0.63 | 2.5 | 0.043/0.074/0.040 | 0.214 (586) |
| v3h T=0.95 | 2 | +0.010 | 0.637 | 0.902 | 0.603 | 0.63 | 2.1 | 0.033/0.033/0.043 | 0.485 (11) |
| v3h T=0.95 | 2 | +0.020 | 0.802 | 0.964 | 0.785 | 0.61 | 10.6 | 0.014/0.008/0.015 | 0.578 (2) |
| v3h T=0.95 | 2 | +0.050 | 0.892 | 0.988 | 0.885 | 0.64 | 2.7 | 0.006/0.002/0.006 | 0.575 (1) |
| v3h T=1.05 | 2 | -0.050 | 0.637 | -0.491 | -0.109 | 0.67 | 3.2 | 0.050/0.004/0.030 | 0.526 (3) |
| v3h T=1.05 | 2 | -0.020 | 0.453 | -0.461 | -0.080 | 0.82 | 3.1 | 0.022/0.019/0.041 | 0.442 (9) |
| v3h T=1.05 | 2 | -0.010 | 0.172 | -0.280 | -0.024 | 0.73 | 3.0 | 0.037/0.045/0.014 | 0.187 (417) |
| v3h T=1.05 | 2 | -0.005 | 0.110 | -0.175 | 0.016 | 0.85 | 24.0 | 0.045/0.348/0.035 | 0.120 (2832) |
| v3h T=1.05 | 2 | -0.002 | 0.106 | -0.087 | 0.023 | 0.72 | 2.0 | 0.011/0.044/0.006 | 0.106 (3829) |
| v3h T=1.05 | 2 | +0.000 | 0.100 | -0.014 | 0.028 | 0.63 | 1.6 | 0.016/0.070/0.014 | 0.104 (4000) |
| v3h T=1.05 | 2 | +0.002 | 0.111 | 0.079 | 0.047 | 0.79 | 2.8 | 0.013/0.062/0.008 | 0.111 (3683) |
| v3h T=1.05 | 2 | +0.005 | 0.133 | 0.234 | 0.080 | 0.67 | 3.3 | 0.020/0.078/0.017 | 0.169 (521) |
| v3h T=1.05 | 2 | +0.010 | 0.401 | 0.707 | 0.358 | 0.67 | 1.7 | 0.032/0.052/0.031 | 0.538 (6) |
| v3h T=1.05 | 2 | +0.020 | 0.730 | 0.957 | 0.710 | 0.68 | 1.5 | 0.011/0.010/0.013 | 0.623 (2) |
| v3h T=1.05 | 2 | +0.050 | 0.846 | 0.986 | 0.838 | 0.70 | 1.0 | 0.005/0.006/0.005 | 0.626 (1) |
| v3h T=1.05 | 4 | -0.010 | 0.193 | -0.372 | -0.028 | 0.75 | 8.7 | 0.083/0.060/0.020 | 0.235 (336) |
| v3h T=1.05 | 4 | +0.010 | 0.432 | 0.727 | 0.396 | 0.77 | 1.7 | 0.069/0.099/0.056 | 0.466 (3) |
| v3h T=1.05 | 4 | +0.050 | 0.854 | 0.987 | 0.848 | 0.51 | 1.6 | 0.006/0.005/0.006 | 0.613 (1) |
| v5a T=1.05 | 2 | -0.050 | 0.664 | -0.493 | -0.096 | 0.80 | 2.8 | 0.027/0.002/0.008 | 0.591 (1) |
| v5a T=1.05 | 2 | -0.020 | 0.425 | -0.457 | -0.081 | 0.61 | 9.2 | 0.075/0.019/0.022 | 0.534 (2) |
| v5a T=1.05 | 2 | -0.010 | 0.215 | -0.329 | -0.019 | 0.70 | 3.5 | 0.023/0.063/0.015 | 0.240 (145) |
| v5a T=1.05 | 2 | -0.005 | 0.156 | -0.187 | 0.009 | 0.63 | 3.4 | 0.017/0.058/0.008 | 0.151 (2418) |
| v5a T=1.05 | 2 | -0.002 | 0.126 | -0.104 | 0.030 | 0.64 | 1.7 | 0.014/0.085/0.011 | 0.131 (3757) |
| v5a T=1.05 | 2 | +0.000 | 0.126 | -0.008 | 0.046 | 0.77 | 3.4 | 0.016/0.046/0.008 | 0.127 (4000) |
| v5a T=1.05 | 2 | +0.002 | 0.127 | 0.052 | 0.048 | 0.83 | 14.7 | 0.073/0.453/0.096 | 0.133 (3639) |
| v5a T=1.05 | 2 | +0.005 | 0.175 | 0.297 | 0.118 | 0.85 | 5.4 | 0.038/0.037/0.025 | 0.180 (881) |
| v5a T=1.05 | 2 | +0.010 | 0.432 | 0.756 | 0.395 | 0.77 | 1.5 | 0.037/0.055/0.028 | 0.451 (10) |
| v5a T=1.05 | 2 | +0.020 | 0.743 | 0.961 | 0.723 | 0.71 | 4.1 | 0.012/0.016/0.018 | 0.611 (2) |
| v5a T=1.05 | 2 | +0.050 | 0.845 | 0.985 | 0.837 | 0.73 | 1.8 | 0.010/0.003/0.011 | 0.635 (1) |
| v5a T=1.05 | 4 | -0.010 | 0.230 | -0.404 | -0.023 | 0.82 | 7.1 | 0.054/0.048/0.020 | 0.255 (509) |
| v5a T=1.05 | 4 | +0.010 | 0.411 | 0.734 | 0.365 | 0.78 | 1.3 | 0.041/0.088/0.043 | 0.334 (63) |
| v5a T=1.05 | 4 | +0.050 | 0.821 | 0.987 | 0.815 | 0.43 | 3.5 | 0.010/0.006/0.014 | 0.572 (1) |

Reading: (i) At small |D| (≤0.005) HMC and reweighting agree (e.g. model B R=2, D=0.005: S 0.175 vs 0.180) and the response is the prior one: S(D=0)=0.10–0.14, χ_eff ≫ MC. (ii) The ensembles switch to a polarised, director-pinned state between D=0.005 and 0.02 (S 0.13 → 0.74), reweighting ESS collapses at the same place (ESS<100 for D ≥ 0.01). (iii) In MC the director is already pinned at D=0.005 (A=0.76) with S nearly unchanged; the VAE ensembles pin later (A=0.2–0.4 at 0.005) because their S is 4× smaller. This is the same prior-hole effect as in E5, not an independent finding about the biaxial channel. (iv) HMC chain-to-chain sd of S, A, B is largest at the transition (D≈0.005–0.01), where τ_int reaches 15–24 for a few points: those points are the least reliable (see the chain-sd column). (v) Easy plane: HMC ⟨B⟩ = −0.019 … −0.031 at D=−0.01, −0.08 … −0.11 at −0.02/−0.05 versus MC +0.013 / +0.006 / −0.003: the VAE biaxial response is >10× MC for |D| ≥ 0.02, i.e. over-estimated (consistent with χ_B,VAE > χ_B,FDT in task A, though much larger than the linear ratio 1.2–1.4 because the VAE ensemble is far from the MC ensemble at these fields).

## Criteria
| criterion | result |
|---|---|
| 1 m=0 reproduces E5 | **met** (model B χ_Q = 0.90–0.93 → ratio 1.00–1.03) |
| 2 H1: χ_B ratio < χ_Q ratio (same model, T, R) | model A: **met** (1.24–1.42 vs 1.67–2.92); model B: **not met** (1.21–1.33 vs 1.00–1.03); "χ_B,VAE ≲ χ_B,site + global share": **not met** (0.26–0.28 vs 0.212 for T=1.05) |
| 3 H2: biaxial response A > B (T=1.05) | **not met** (equal within noise) |
| 4 non-linear: HMC reproduces MC S(D), ⟨P₂(d·e)⟩; easy-plane ⟨B⟩ under-estimated | **not met** (prior-hole offset of S, delayed director pinning; easy-plane ⟨B⟩ over-estimated >10×, not under-estimated) |
| 5 collective biaxial part mainly from director rotation subspace → correct H1 | **not the case**: the part comes from the global latent (cell deviations contribute ≲0.05); the H1 reading "channel information G" is neither confirmed nor corrected by this — the excess is a global-latent effect |

## Caveats
(i) χ_B FDT references are those of `fdt_biax.json` (not recomputed); their collective part also contains the second-order director effect noted in the request. (ii) H2 comparison uses the aggregated posterior; on the prior the two models differ by <25 % (χ_B,lat 3.7–4.3 vs 4.6–4.9) in the opposite sense. (iii) The "global only" decomposition removes deviations but leaves the global latent's own influence on the cell channels; it is an operational split, not an additive one (parts do not sum to the total). (iv) HMC for R=4 and for T=0.95 is partial; R=3 missing. (v) The easy-plane ⟨B⟩ observable is defined with e'=d×e, which is only a biaxiality measure for e ⟂ d; for D≥0 and at D=0 its value is dominated by the orientation. (vi) E5 Fig. 1 fixes (legend names; MC dashed line no longer flat below h=0.02, now m=h/6 on a log grid) were made in `make_E5.py` and the figure regenerated.
