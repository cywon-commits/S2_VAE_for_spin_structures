| T | decoder | KL | mean term (1/2 sum mu^2) | variance term | NLL | ELBO | active l=2 channels: |mu| | sigma |
|---|---|---|---|---|---|---|---|---|
| 0.95 | v2 pruned (baseline, trained 40k+10k) | 28.2 | - | - | 1.4816 | -2588.4 | - | - |
| 0.95 | plain decoder re-initialised (control, 25k) | 26.6 | 16.3 | 10.3 | 1.4819 | -2587.2 | [5.71, 0.05] | [0.078, 1.007] |
| 0.95 | direction/magnitude separated, all active l=2 channels (25k) | 15.1 | 2.4 | 12.7 | 1.4819 | -2575.8 | [2.19, 0.15] | [0.048, 0.998] |
| 1.05 | v2 pruned (baseline, trained 40k+10k) | 20.3 | - | - | 1.8120 | -3151.5 | - | - |
| 1.05 | plain decoder re-initialised (control, 25k) | 22.9 | 12.6 | 10.3 | 1.8122 | -3154.4 | [5.01, 0.07] | [0.077, 1.0] |
| 1.05 | direction/magnitude separated, all active l=2 channels (25k) | 14.0 | 2.4 | 11.7 | 1.8122 | -3145.6 | [0.08, 2.16] | [0.998, 0.059] |
