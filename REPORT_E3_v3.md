# REPORT_E3_v3 — oracle colouring test (channel-locking hypothesis)

Follows `REPORT_E3_v2.md`; request `REQUEST_E3_KAGOME_v3.md`. Same data and split (validation = 8 seeds, "±" = seed s.e.; regressions seed-clustered). One training run per cell.

## Verdict
**The channel-locking prediction was not confirmed with this implementation.** Feeding the oracle colour of every site (rotation-invariant one-hot) to the R=4 site-resolved decoder did *not* make decoded site distributions single-cap (mean-vector length 0.13, prediction ≥0.8), did *not* raise the regression coefficient a (0.105±0.066, was 0.108), and its NLL gain over the colour-free model is indistinguishable from the shuffled-colour control. Whether this means the hypothesis is wrong or that this particular way of supplying colours cannot work is **not established** (see "Why it may fail" below).

## Setup (A)
- Colours: per snapshot, `global_triad` of `kagome/loops.py`, c_i = argmax_k n̂_i·u_k (computed separately for the time-reversed training copies). One-hot (3) enters the decoder only through the per-site mixing weights of the K=12 cell-level multipole sets: w_pk = emb[p,k] + onehot(c_p)·Wc[:,k] (low-rank, rotation invariant). Encoder, latent, prior and KL unchanged; colours are not in the KL.
- Training: fine-tune of the v2 models (same encoder/decoder weights, new Wc), 6000 steps, lr 7e-4. The shuffled-colour control (colour labels randomly permuted among the sites of each snapshot, fixed seed) went through exactly the same fine-tuning, so oracle−shuffled isolates the colour information, while oracle−(v2) also contains the effect of extra training.
- Colour information content (requested separately): with the 3-colouring given, the upper bound on the information supplied is ln 3 = 1.099 nats/site; the reference lower value quoted in the request is 0.13. I did not estimate the actual entropy of the colouring.
- Not run: (iv) R=12 + colours (optional), the R=1 shuffled control, T=0.001/0.004, and task B (optional; no resources spent).

## Results
| model | T | NLL/site | KL | net gain vs R=1 (nats/cfg) | NLL gain vs colour-free v2 | decoded mean-vector length | decoded θ² (rad²) | data colour-class mean length / θ² |
|---|---|---|---|---|---|---|---|---|
| (i) R=4, no colour (v2) | 0.0005 | -0.557 | 234.8 | 97.7 ± 3.4 | — | ≈0.14 (v2 report) | — | — |
| (i) R=4, no colour (v2) | 0.002 | 0.427 | 178.9 | 214.2 ± 8.9 | — | ≈0.14 (v2 report) | — | — |
| (ii) R=4 + oracle colours | 0.0005 | -0.578 | 229.0 | 112.7 ± 3.2 | 0.0215 ± 0.0007 | 0.132 | 2.60 | 0.953 / 0.10 |
| control: R=4 + shuffled colours | 0.0005 | -0.576 | 228.1 | 112.7 ± 3.2 | 0.0192 ± 0.0008 | 0.127 | 2.62 | 0.261 / 2.11 |
| (iii) R=1 + oracle colours | 0.0005 | 0.037 | 59.6 | 16.6 ± 4.1 | 0.0823 ± 0.0066 | 0.211 | 2.34 | 0.937 / 0.13 |
| (ii) R=4 + oracle colours | 0.002 | 0.410 | 179.3 | 221.2 ± 8.3 | 0.0170 ± 0.0009 | 0.054 | 2.90 | 0.906 / 0.20 |

(Decoded quantities use the posterior-mean decode. "Data colour class" = spins of one oracle colour inside one cell, mean-vector length and mean squared angle to their own mean; in the shuffled control the classes are random mixtures, which is why they look broad.)

Observations:
1. **Data side supports single caps per colour:** with oracle colours the spins of a colour inside a cell have mean-vector length 0.95 (T=0.0005), 0.91 (T=0.002) and θ² ≈ 0.10–0.20 rad², whereas the shuffled classes are broad (0.26, θ² 2.1). So the colouring itself is informative and consistent with the premise that a colour-resolved distribution is a single cap.
2. **The decoder does not exploit it:** decoded mean-vector length 0.13 (T=0.0005), 0.054 (T=0.002), 0.21 (R=1); decoded θ² ≈ 2.3–2.9 rad², i.e. as broad as the colour-free model.
3. **Oracle vs control (R=4, T=0.0005):** NLL −0.5781 vs −0.5758; net gain over v2 +15.0±0.3 vs +15.0±0.4; NLL gain over colour-free 0.0215±0.0007 vs 0.0192±0.0008. The difference oracle−shuffled is ≈0.002 nats/site (≈1 nat per snapshot): essentially nothing. The +15 nats over v2 is mostly the effect of the additional fine-tuning (present in the control).
4. The R=1 oracle model improved by 33.6±2.8 nats (NLL 0.082 nats/site) over its colour-free version, but **I have no shuffled control for R=1**, so this cannot be attributed to the colours.
5. Prediction 3 (net gain growing by ≈5 nats per ring × rings per snapshot, i.e. of order 200 nats at T=0.0005, within a factor 2): **not met** (colour-attributable gain ≈1 nat at R=4).
6. Prediction 4 (colour-resolved cell-frame harmonic ratios): not computed; the decoded distributions are not colour-resolved single caps, so there is nothing to compare.

## Regression (C)
| model | T | target y | a ± se | b ± se | c ± se | R² |
|---|---|---|---|---|---|---|
| R=4 oracle | 0.0005 | N[NLL(R=1)−NLL] | 0.105 ± 0.066 | — | 276.9 ± 9.8 | 0.01 |
| R=4 oracle | 0.0005 | N[NLL(colour-free)−NLL] | -0.003 ± 0.004 | — | 9.6 ± 0.6 | 0.00 |
| R=4 shuffled | 0.0005 | N[NLL(R=1)−NLL] | 0.119 ± 0.066 | — | 274.0 ± 9.9 | 0.01 |
| R=4 shuffled | 0.0005 | N[NLL(colour-free)−NLL] | 0.012 ± 0.002 | — | 6.7 ± 0.5 | 0.04 |
| R=4 oracle | 0.002 | N[NLL(R=1)−NLL] | -0.109 ± 0.072 | 16.0 ± 1.3 | 355.5 ± 10.6 | 0.30 |
| R=4 oracle | 0.002 | N[NLL(colour-free)−NLL] | 0.008 ± 0.007 | -0.2 ± 0.1 | 6.7 ± 1.0 | 0.02 |

Prediction a≈1: not met in any row; a stays at 0.1 (T=0.0005) / −0.1 (T=0.002) against the R=1 baseline, and against the colour-free model a ≈ 0 (±0.004–0.007) for oracle and 0.012±0.002 for the shuffled control at T=0.0005 (a small but non-zero value in the control, which I cannot explain; it is a different fine-tuning trajectory, not a colour effect). The constant c remains large (≈277 and ≈356 nats) and the defect coefficient is unchanged (b = 16.0±1.3 nats per defect at T=0.002).

## Why it may fail (not tested)
- **Label gauge.** Colour indices from `global_triad` are defined by an eigenvector sign and a phase branch (arg⟨e^{3iφ}⟩/3 is defined modulo 2π/3), so the mapping "colour k → cap direction" is an arbitrary cyclic relabelling from snapshot to snapshot. The latent's octupole (l=3) content is symmetric under that cyclic relabelling, so the decoder cannot tell from it which of the three caps colour 0 refers to; only an l=1 latent (or other symmetry-breaking content) could carry that label, and the latent power in l=1 is ≈0 (0.5–5%) and it costs KL. If so, the colour-conditioned decoder would learn nothing from the oracle colours *by construction*, independently of the validity of the locking hypothesis. A fair test needs colours defined relative to something the decoder can see (e.g. colours assigned relative to the decoder's own decoded cap triad, or a colour-resolved latent, or the encoder given the colour labels).
- **Capacity of the colour pathway.** Colours act only through low-rank mixing weights, not multiplicatively inside the tensor products; a cap-selecting nonlinearity may be hard to learn in 6000 fine-tuning steps from a colour-free solution.
- I tested neither of these; the evidence only shows that this implementation yields no colour benefit.
