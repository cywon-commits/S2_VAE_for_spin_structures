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
@@T1@@

Columns: "ch.1/ch.2" are the two active l=2 channels (components 21–25 and 31–35; the second is the main one, `act`=31). Prior: z_g ~ N(0,I) (E‖z‖²=5, variances 1); cell noise ε: N(0,I). Model B at R=3, 4 has a collapsed first channel (variance 0.01–0.04) – the posterior there lives only in the main channel.

S of ancestral samples (sample frame, per z; mean ± sd over samples; data = validation snapshots):

@@T2@@

(i)/(ii) are the decompositions requested in 1-a; the data S sd is 0.022 (var 4.9×10⁻⁴).

## Task 1-b — refit and linear check
Mixture: z_g posterior samples (active 10 components, 800 configs × 8 draws) × 16 random SO(3) rotations (the same D² Wigner matrix on both l=2 channels), full-covariance Gaussian mixtures, k ∈ {50, 100, 200} chosen by BIC (the request asked 20–50; BIC picked the smallest k=50 in every case, which is the lower BIC but not the best S width — see below), reg_covar 10⁻⁴. Cells: conditional prior as before (z_c = z_g + τε; no cell-level refit needed). Figure `E7_fig_Sdist`.

@@T3@@

The refit lowers the old prior's χ (χ_Q 12.6–23, χ_B 4.0–6.4) to χ_Q 2.7–3.9 and χ_B 0.19–0.31 (aggregated posterior: χ_Q 0.9–1.6, χ_B 0.18–0.27; MC 0.90/0.39 and 0.212/0.129). The biaxial value is therefore repaired (model B 0.220–0.264 vs posterior 0.256–0.272; model A 0.28–0.31 vs 0.26–0.27 at T=1.05, 0.19–0.20 vs 0.18 at T=0.95), the amplitude value remains 2–4× too large (Verdict 3). Criterion "S mean within 0.02 of data": met for B R=2 (0.002), B R=3 (0.010), A T=0.95 (0.010–0.016), A T=1.05 R=2 (0.016); **not met** for B R=4 (0.024) and A T=1.05 R=3, 4 (0.038, 0.033; for model A the conditional-prior cells alone, case (i), already lower S by 0.01–0.036). "Variance ratio 0.7–1.4": **not met** (3.4–7.4).

## Task 1-c — non-linear response with the refitted prior (T=1.05)
HMC in the non-centred coordinates (z_g from the mixture, ε~N(0,I)); 12 chains, 100+150 steps (R=4: 8 chains, 40+60 steps, reduced D grid); MC = E6 task C.

@@T4@@

Criteria (T=1.05; slopes dS/dD from D = 0.01 to 0.05, MC 0.65; easy plane dS/d|D| from |D| = 0.01 to 0.05, MC 0.38):
- (i) S(D=0) within 0.02 of data (0.558): B R=2 0.560 ✓; A R=2 0.540 (0.018 ✓, its ancestral S is also low); B R=4 0.533 (0.025 ✗ — its own posterior S is 0.535).
- (ii) pinning at D=0.005, ⟨P₂(d·e)⟩ ≥ 0.7: B R=2 0.755, B R=4 0.780, A R=2 0.763 (MC 0.760) ✓.
- (iii) slope in the pinned window within 20 % of MC: **✗** — easy axis 4.97 (B R=2), 5.2 (B R=4), 5.3 (A R=2) vs 0.65 (k=200 mixture, B R=2: 4.2); easy plane 2.53 (B R=2) vs 0.38. The VAE is right for D ≤ 0.01 (S within 0.01–0.02 of MC, e.g. B R=2 D=0.01: 0.603 vs 0.568, D=−0.01: 0.575 vs 0.564) and departs at |D| ≥ 0.02 (0.60–0.80 vs 0.57–0.59).
- (iv) easy-plane ⟨B⟩ at the MC size: ✓ for |D| ≥ 0.01 (|⟨B⟩_VAE − ⟨B⟩_MC| ≤ 0.011, B R=2 0.016 / 0.007 / −0.006 vs 0.013 / 0.006 / −0.003; E6: −0.08…−0.11). At |D| = 0.002–0.005 the VAE is 0.016–0.03 above MC (director pinning is slightly softer: ⟨P₂(d·e)⟩(−0.002) = −0.24 (B), −0.26 (A) vs −0.33 in MC; at D=+0.002 0.39–0.54 vs 0.27), consistent with its larger S sd and with chain-to-chain sd up to 0.2 (A) at D=0.
The remaining non-linear discrepancy is therefore confined to S at |D| ≥ 0.02, i.e. the amplitude channel, the same channel that overestimates χ_Q,lat (Verdict 3).

## Task 2 — wave-vector resolved response χ(q)
MC: `data/ll_L12_T{0.950,1.050}_o.npy` (2000 snapshots; frame rule of `fdt_biax.py`); reproduces χ_Q(0) = 0.390 / 0.903 and χ_B(0) = 0.129 / 0.212. VAE: χ_dec (q-independent) + χ_lat(q), χ_lat(q)=(β/N) Var_z|Σ_i⟨O_i⟩_z e^{iq·r_i}|² from the cell-constant decoder means expanded onto the 12³ lattice (aggregated posterior, validation × 4 draws), shells in |k|² (q = 2π k/12). Prediction: χ_site + (χ_MC(k) − χ_site)|F_b(k)|² per k vector, then shell-averaged. Figures: `figures/E7_fig_chiq_T1.05.*` (model B and A, three channels), `E7_fig_chiq_T0.95.*` (model A). "Excess ratio" = (χ_VAE − χ_site)/(χ_pred − χ_site) averaged over the shells with 0 < k ≤ R/2 (low) and R/2 < k < R (mid; k=R is the cell wave number 2π/b, where F_b = 0).

@@T5@@

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
