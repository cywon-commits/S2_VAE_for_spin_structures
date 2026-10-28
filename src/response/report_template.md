# REPORT_E5 — field response of the H=0-trained VAEs versus Monte Carlo

Request `REQUEST_E5_RESPONSE_v1.md`. No new training: existing checkpoints only. All numbers are per site, J=1; "±" = standard error over the stated samples. Code: `src/response/` (library `vae_response.py` (kagome), `nem_response.py` (nematic); drivers `run_A_kag.py`, `run_A_nem.py`, `run_B_kag.py`, `run_B_mcmc.py`, `run_B_nem.py`, `ll_quad_mc.py`, `ll_quad_fine.py`, `nem_check.py`, `make_E5.py`, `tables_E5.py`); raw output `results/response/`; figures `figures/E5/`.

## Verdict
1. **The sign of the prediction holds, its size does not.** For every kagome checkpoint the VAE linear susceptibility is *larger* than the MC fluctuation value (χ_VAE ≥ χ_FDT), as predicted for a decoder that discards within-cell correlations. But the excess is not the "1.5–1.7 × χ_W" the request anticipates: on the aggregated posterior χ_VAE/χ_FDT is **≈ 7×10³ (R=1) → 2×10³ (R=12) at T=0.0005** and **≈ 1.3×10³ → 5×10² at T=0.002** (in-plane); the best control (single-cap decoder with oracle colour input) is still 1.8×10² / 1.3×10² times too large. The decoder part alone (χ_dec = β·E Var(n|z), the sites treated as independent) is 23–900 versus the *conditional* Weiss value χ_W = 0.27/0.45; i.e. what the independent-site decoder discards is not a small correction but the triangle (Σ spins ≈ 0) constraint itself.
2. **Consequently m(h) from the VAEs is wrong already at h ~ 10⁻⁵.** Exact importance reweighting of prior samples has an ESS≥100 window of only h ≲ 1×10⁻⁵ at T=0.0005 (3×10⁻⁵–10⁻⁴ at T=0.002; MC window: 0.011); inside it m/h = 2.4×10³–7.3×10³ (true 1/6, exactly the analytic prior χ). Latent-space HMC (acceptance 0.6–0.8, chains agree to <0.5 %) shows the VAE ensemble is **already m ≥ 0.96 at h=10⁻⁴ and m ≥ 0.999 for h ≥ 0.02**, where MC has m = h/6 = 0.003…0.33. This is a model failure, not a sampler failure.
3. **Nematic Zeeman: pass** (χ_VAE = Curie within 1 % on the aggregated posterior). **Nematic quadrupole:** decoder part χ_dec = 0.10–0.14 ≈ χ_W (0.087–0.113); the latent part restores the order-parameter fluctuations. χ_VAE/χ_FDT = 2.7/2.8/2.7 (v3h, T=0.95, R=2/3/4), 1.7/1.7/1.7 (v3h, T=1.05) and **0.98/1.03/1.01 (v5a, T=1.05)**: the sign criterion (χ_VAE < χ_FDT) is *not* met by any model; only v5a reproduces χ_FDT (within its 5 % s.e.).
4. **Prior vs aggregated posterior differ by huge factors** (the prior samples are mostly disordered: S_model = 0.10–0.17 versus 0.56–0.66 in the data; for kagome the decoded net moment of prior samples is unbalanced). All verdicts above are on the aggregated posterior (as the request prescribes when the difference is large); prior values are listed in the tables.
5. **Direction selection** (⟨(ν·ĥ)²⟩ → 0 at h ≳ 0.5 as in MC): VAE gives 0.00–0.01 for h ≥ 10⁻³, but this is not evidence of correct physics — see caveat (ii).

## Method (what was actually computed)
- *Decoder = exponential family* p(n_i|z) ∝ exp(λ_i·Y(n)); a Zeeman term βh n·ĥ adds Δλ_{1m}=βh(4π/3)Y_{1m}(ĥ) (integral-normalised Y), a quadrupole term βε P₂(n·d) adds Δλ_{2m}=βε(4π/5)Y_{2m}(d). Site moments / log Z come from Lebedev quadrature (kagome: order 131, l≤8; nematic: order 59, l≤6 — the quadrature of the trained models). Sites are conditionally independent given z (also for the cell-piecewise-constant decoders: the (cell,sublattice) λ is shared by (12/R)² (kagome) or (12/R)³ (nematic) sites, each an independent draw).
- *Linear response* χ_ab = (β/N)[E_z Σ_i Cov_i(O_a,O_b|z) + Cov_z(Σ_i⟨O_i⟩_z)] = χ_dec + χ_lat, evaluated analytically per z in the **sample frame** (kagome: ν(z) = lowest eigenvector of Σ_i⟨n_i n_iᵀ⟩_z, in-plane = mean of the two in-plane diagonals; nematic: director d(z) = top eigenvector of Σ_i⟨n_i n_iᵀ⟩_z; quadrupole operator P₂(n·d(z))). ⟨M̄⟩=0 by isotropy. (Frame from second moments, not from ⟨n_i⟩, because the colour-free kagome decoders give three-cap site mixtures with ⟨n_i⟩≈0.)
- *Latents*: ancestral prior samples (K=2000; hierarchical priors top-down: z_g~N(0,I), z_c=z_g+τ_l ε, for MVAE8 main l=2 channel with rotation/amplitude subspace deviations, biaxial part tied to z_g, subspaces from the sampled z_g — my reading of the v5a prior) **and** aggregated posterior z~q(z|n_data) on the validation data (kagome: 800 validation snapshots × 4 draws; nematic: validation split × 8 draws). `kag6u … oracle` needs per-site oracle colour vectors from the data and is therefore evaluated on the posterior only; for `nolatent` the colour vectors u come from random validation snapshots, so its "lat" column is the variance of the net moment across the *u-field* (the latent itself is unused).
- *Check 1* (generated batch): spins drawn from the decoder, (β/N)⟨M_a M_b⟩ in the realised sample frame (kagome, 60 z × 5 draws) or βN Var(S_p) with the realised director (nematic, 1000 z × 2). Compared with the analytic value on the *same* 60 z, generated/analytic = 0.88–1.14 for all 18 kagome checkpoints (in-plane and normal; `A_kag.json` keys `posterior_check` vs `posterior_analytic_first`); nematic: v3h T=0.95 posterior 1.06±0.05 vs analytic 1.05 (`A_nem_check.json`). The table's last column is the generated value on 60 z, not on the full 3200 draws.
- *Task B (reweighting)*: p_θ^H ∝ p(z) Π_i Z(λ_i+βH·n)/Z(λ_i) · p(n_i|λ_i+βH·n); weights computed in log space, self-normalised, ESS=(Σw)²/Σw²; lab-fixed field with an independent random direction ĥ per z (equivalent to the random global rotation); m(h)=⟨Σ_i n_i·ĥ⟩/N. Because the random ĥ gives a sampling bias of m at h=0 (≈3×10⁻³ for K=2000), the h=0 estimate is subtracted *only where ESS≥100*. Beyond ESS collapse: HMC on (z_g, z_c) with the joint prior density and log w(z;h) (autograd through decoder and log Z), field along e_z (prior is rotation invariant), 16 chains, 15 leapfrog steps, step size adapted to 0.7 acceptance in warm-up.
- *Task C*: ll_mc.py heat bath with the extra Bingham matrix (3/2)βε d dᵀ, d=ẑ, L=12, 4 seeds, ordered start, 400 equilibration + 2000 measurement sweeps (every 5th measured).

## Task A — linear susceptibility
@@TABLES@@

Notes: kagome columns are "in-plane / normal (ν)". χ_FDT: 40-seed MC fluctuation values (`fdt_kag.json`), χ_W: frozen-Weiss. "χ_dec/χ_lat" for the posterior use the 800×4 aggregated-posterior draws, for the prior K=2000.
- χ_VAE/χ_FDT versus R (aggregated posterior, in-plane): T=0.0005: 7607, 7034 (R=3), 6562 (R=4), 5797 (R=6), 2004 (R=12), R=1 with colour-free tied prior (`kag6`) 6941; T=0.002: 1309, 1290, 1264, 1158, 482. So χ_VAE − 1/6 **does decrease with R** (R=12 is 2.7–3.8× lower than R=1) but stays 2–4 orders above the truth (Fig. 2a).
- Anisotropy (χ_normal/χ_in-plane, aggregated posterior): 0.16–0.29 (T=0.0005) and 0.42–0.58 (T=0.002) for all colour-free models (rising with R), i.e. in-plane ≫ normal, opposite to the Weiss expectation 1.5–1.7 (truth: 1.00 / 0.96); for the controls 1.39 (nolatent, T=0.0005), 1.04 (T=0.002) — inside the 1–1.7 band although the magnitudes are wrong — and 4.8 / 3.9 for the oracle model. The criterion is **not met** by the physical (colour-free) models.
- Controls: `kag6u nolatent` (single-cap decoder from the data colours, no latent) has χ_dec = 66.9/106 (T=0.0005) and 30.3/47.7 (T=0.002) — 250–400× χ_W, so "χ_dec ≈ χ_W" **fails**. Reason: the decoder's single-site angular spread (σ² ≈ 0.05–0.1 per component, a *marginal* variance) is used as an independent Curie-like fluctuation, χ_dec ≈ βσ², while χ_W is the *conditional* variance given the neighbours (≈10⁻⁴ at T=5×10⁻⁴). The oracle model is better in-plane (22.7) but not in the normal direction (136).
- Nematic Zeeman (aggregated posterior): χ_∥/χ_⊥ = 0.816/0.118 (T=0.95), 0.670/0.141 (T=1.05) vs Curie 0.817/0.118, 0.673/0.140 → deviation ≤1 %; the latent contribution to Zeeman is 10⁻⁴ (⟨n_i⟩≈0 for the even decoders). Criterion met. (On prior samples the Zeeman value is wrong: 0.44/0.30, because of the prior's disordered samples.)
- Nematic quadrupole, aggregated posterior: χ_dec ≈ 0.10–0.14 ≈ χ_W = 0.087–0.113 (ratio 1.1–1.3); χ_lat (variance of the director-frame order parameter across configurations, scaled) = 1.02–1.08 (v3h T=0.95), 1.41–1.43 (v3h T=1.05), 0.76–0.81 (v5a T=1.05). χ_FDT − χ_dec = 0.32 (T=0.95), 0.78 (T=1.05): v5a matches the latter (0.76–0.81), v3h overshoots by 3× (T=0.95) / 1.8× (T=1.05). R-dependence is weak (R=2–4 within 5 %). T-dependence of χ_VAE/χ_FDT for v3h: 2.7 → 1.7 (T=0.95 → 1.05); only T=0.95, 1.05 checkpoints exist (no 0.80/1.10).

## Task B — m(h)
### B1. Importance reweighting of prior samples (K=2000; T=0.0005)
@@RW@@

m_eff = m/h inside the window equals the analytic prior χ (isotropic average (2χ_in+χ_ν)/3 of the prior row, e.g. R=1: 3.9×10³ vs 4.06×10³). Inside the window ⟨(ν·ĥ)²⟩ falls below 1/3 (0.317, 0.293, 0.330 at h=10⁻⁵ for R=1, R=4, kag6 R=4), i.e. the h² coefficient of the orientation distribution is negative: the field prefers to lie *in* the plane, as in MC (MC: 0.23 at h=0.1 → 0 at h≳0.5), but with the opposite sign relative to the frozen-Weiss expectation (which would select ν∥ĥ) because the VAE's in-plane response is the larger one. Above the window, ESS collapses to 1–5 and m saturates (dotted curves in Fig. 1). For nematic quadrupole reweighting (`B_nem.json`, K=4000, prior samples, random d per z): window ε ≲ 0.002–0.005 (ESS ≥ 500), within which S(ε)/ε ≈ 13 (v3h, T=0.95/1.05) — dominated by the prior's disordered samples (χ_prior ≈ 20, 11) and ≫ MC.

### B2. Latent-space HMC (T=0.0005)
@@MC@@

Interpretation: after 150 warm-up steps the 16 chains agree to <0.5 % (chain-to-chain sd column), acceptance 0.6–0.8, τ_int≈1–9: the sampler works and finds a fully polarised ensemble. m is within ≈4 % of saturation at h=10⁻⁴ (MC: 1.7×10⁻⁵). (ii) At full polarisation the decoded second-moment matrix has rank 1, so the "plane normal" ν is not defined and ⟨(ν·ĥ)²⟩≈0 is a trivial artefact: **do not read h ≥ 10⁻³ ν-values as successful direction selection.**

## Task C — quadrupolar field in the LL MC
@@CT@@

S(ε) is strongly concave: the director is pinned only for Nε S ≳ T (ε ≳ 2×10⁻³; S(0)=0.15–0.26±0.1 with a free director), then S grows slowly. The "linear" slope depends on the window: ε=0.02–0.1 gives 0.53/0.77/1.20 against χ_FDT = 0.42/0.91/2.4 (T=0.95/1.05/1.10); the ratios 1.3/0.85/0.5 show agreement within a factor 2 at T ≤ 1.05; at T=1.10 (close to T_c≈1.12) no linear window exists on this grid (S(0.2)=0.61 still rising), and the finite-size director fluctuation adds an extra ~T/(NεS) ≈ 1 % rise in S between ε=0.01 and 0.02 which inflates the slope in that window. I therefore treat FDT and MC as consistent but the MC slope as a soft check, not a tight one.

## Criteria (§5 of the request)
| criterion | result |
|---|---|
| nematic Zeeman χ = Curie ± 2 % | **met** (posterior, ≤1 %) |
| kagome χ_FDT ≤ χ_VAE | met (all 18 kagome checkpoints), but by 10²–10⁴×, not 1.5×–few × |
| χ_⊥/χ_∥ between 1 and 1.5–1.7 | **not met** for colour-free models (0.16–0.58, opposite direction); the nolatent control gives 1.04–1.39 (in band) but with 100× too large χ |
| nolatent control: χ_lat ≈ 0, χ_dec ≈ χ_W | **not met** (χ_dec 30–107 vs 0.27–0.45; "lat" is u-field variance 9–16 in-plane) |
| χ_VAE − 1/6 decreasing with R | met (monotone; R=1→12 reduces χ by 3.8× (T=0.0005) and 2.7× (T=0.002)) but never approaches 0 |
| nematic quadrupole χ_VAE < χ_FDT | **not met** (1.7–2.8× for v3h; 1.0× for v5a) |
| nematic χ_lat ≈ χ_FDT − χ_dec | met for v5a (0.76–0.81 vs 0.78), not for v3h |
| direction selection ⟨(ν·ĥ)²⟩ → 0 for h ≳ 0.5 | formally 0.00, but artefact of full polarisation (caveat ii) |
| prior vs aggregated posterior | reported separately; kagome prior/posterior in-plane χ_VAE ratio 1.6–28×; nematic quadrupole χ_VAE/χ_FDT: prior 12–50 vs posterior 1.0–2.8 |

## Caveats / honest limits
(i) MC (T/J=0.001, 0.005) and VAE (0.0005, 0.002) temperatures do not coincide; the reference χ_FDT values are at the VAE temperatures, the m(h) curves of Fig. 1 are plotted against the MC points at 0.001/0.005 (the MC curve is T-independent to the plotted precision).
(ii) Full-polarisation artefact of ν (above).
(iii) The sample-frame ν(z) is defined by decoded second moments; per-site means ⟨n_i⟩ were not used because they vanish for three-cap mixtures.
(iv) The prior used for MVAE8 (subspaces from the sampled z_g) is my reading of the model; the aggregated-posterior numbers do not depend on it, the prior numbers do.
(v) K=2000 prior samples: χ_lat is heavy-tailed (relative s.e. up to ≈30 % in prior R=1, ≈7 % elsewhere); reported in `A_kag.json` (`*_se`).
(vi) HMC for R=4 used a reduced budget (60 warm-up/100 sampling steps, 8 chains, 4 h values); R=12, the kag6 models and nematic latent-space HMC were not run. Reweighting at T=0.002 and R=12 is included only where listed in the tables above.
(vii) Fig. 1 solid/dotted split follows ESS≥100; the HMC points (×) are shown without ESS since they are not importance-sampled.
(viii) Task C was run on ordered starts with 4 seeds only; the S(0) values (free director) are noisy by construction.

## Figures
`figures/E5/fig1_kagome_mh.png` (m(h): MC vs VAE reweighting, dotted where ESS<100, HMC points ×, linear-window panel), `figures/E5/fig2_ratio.png` (χ_VAE/χ_FDT vs R for the two T; nematic quadrupole ratio vs T), `figures/E5/fig3_nu_h2.png` (⟨(ν·ĥ)²⟩ vs h).
