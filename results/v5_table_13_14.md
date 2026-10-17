| R | model | NLL gain vs R=1 | extra KL vs R=1 | net ELBO gain vs R=1 | predicted net ELBO | extra KL / (dir+amp budget) |
|---|---|---|---|---|---|---|
| 2 | isotropic up/down (v4) | 0.042 | 48 | +24 | +42 | 2.3x |
| 2 | subspace covariance (v4) | 0.043 | 36 | +39 | +42 | 1.7x |
| 2 | 1.3 ablation: no biaxial deviation | 0.039 | 28 | +38 | +42 | 1.3x |
| 2 | 1.4 separated decoder + subspace | 0.042 | 30 | +43 | +42 | 1.4x |
| 3 | isotropic up/down (v4) | 0.105 | 130 | +52 | +101 | 2.1x |
| 3 | subspace covariance (v4) | 0.109 | 111 | +77 | +101 | 1.8x |
| 3 | 1.3 ablation: no biaxial deviation | 0.096 | 90 | +76 | +101 | 1.4x |
| 3 | 1.4 separated decoder + subspace | 0.106 | 89 | +95 | +101 | 1.4x |
| 4 | isotropic up/down (v4) | 0.176 | 252 | +53 | +140 | 2.0x |
| 4 | subspace covariance (v4) | 0.184 | 226 | +91 | +140 | 1.8x |
| 4 | 1.3 ablation: no biaxial deviation | 0.159 | 187 | +89 | +140 | 1.5x |
| 4 | 1.4 separated decoder + subspace | 0.177 | 182 | +124 | +140 | 1.5x |

(R=1 baseline: NLL 1.8120, KL 20.3, ELBO -3151.5; separated-decoder R=1 model: NLL 1.8122, KL 14.0, ELBO -3145.6)
