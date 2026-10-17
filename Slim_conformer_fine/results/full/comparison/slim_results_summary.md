# Slim Conformer B — 28-pair results

Slim Conformer B (25,752 parameters) against Conformer A and Conformer B from
`Conformer_decimated_CWT/results/full`. FineMI 0.5-3 Hz, decimated 250 -> 50 Hz (D = 5),
within-subject 5-fold CV, 18 subjects, 50 epochs, seed 42. Same folds, seeds and
training as the reference runs, so every subject pairs across the three models.

Run: 28 pairs on 2 x NVIDIA A100-SXM4-40GB (Vast.ai), torch 2.8.0+cu128, 9.0 GPU-hours
(19.3 min per pair), about 4.5 h of wall time.

## 1. Pooled accuracy (%), 28 pairs x 18 subjects

Each subject's accuracy is averaged over the 28 pairs first, then averaged over subjects.
Chance is 50%.

| Model | Parameters | 800 ms | 1500 ms | 3000 ms | 4000 ms | Average |
|---|---:|---:|---:|---:|---:|---:|
| Slim Conformer B | 25,752 | 61.78 | 64.80 | 64.40 | 63.94 | 63.73 |
| Conformer A | 153,978 - 221,818 | 63.05 | 66.61 | 67.96 | 67.80 | 66.35 |
| Conformer B | 260,034 | 63.83 | 66.68 | 66.46 | 66.11 | 65.77 |

Conformer A's parameter count grows with the window: 153,978 / 169,338 / 201,338 / 221,818.

## 2. Paired differences (points), Slim minus reference

Paired t-tests across 18 subjects, Holm-corrected within each scope. Every cell below
is significant at p < .001 after correction. W = subjects where Slim is better / worse.

| Contrast | 800 ms | 1500 ms | 3000 ms | 4000 ms | Average | 95% CI on the average | W (average) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Slim - Conformer A | -1.27 | -1.81 | -3.55 | -3.86 | -2.62 | [-3.18, -2.07] | 0 - 18 |
| Slim - Conformer B | -2.06 | -1.87 | -2.06 | -2.16 | -2.04 | [-2.40, -1.68] | 0 - 18 |

Verdict at a non-inferiority margin of 1.0 point on the window average: Slim is WORSE
than both references. The whole confidence interval sits below -1 in each case.

## 3. Per pair, window average (%)

| Pair | Slim | Conformer A | Conformer B | Slim - A | Slim - B |
|---|---:|---:|---:|---:|---:|
| HOC / WFE | 58.87 | 61.10 | 60.17 | -2.24 | -1.30 |
| HOC / WAA | 60.91 | 63.97 | 64.95 | -3.06 | -4.04 |
| HOC / EPS | 60.30 | 62.66 | 62.57 | -2.37 | -2.27 |
| HOC / EFE | 64.41 | 64.64 | 66.73 | -0.23 | -2.32 |
| HOC / SPS | 68.41 | 69.79 | 69.49 | -1.38 | -1.08 |
| HOC / SAA | 70.63 | 72.74 | 72.48 | -2.11 | -1.86 |
| HOC / SFE | 67.56 | 69.59 | 70.34 | -2.04 | -2.78 |
| WFE / WAA | 60.96 | 61.94 | 62.45 | -0.98 | -1.49 |
| WFE / EPS | 59.65 | 61.86 | 61.22 | -2.20 | -1.56 |
| WFE / EFE | 60.64 | 62.71 | 62.25 | -2.07 | -1.60 |
| WFE / SPS | 68.02 | 69.31 | 69.05 | -1.29 | -1.02 |
| WFE / SAA | 66.05 | 69.26 | 67.50 | -3.22 | -1.46 |
| WFE / SFE | 61.75 | 65.34 | 64.26 | -3.58 | -2.51 |
| WAA / EPS | 58.39 | 61.54 | 60.37 | -3.15 | -1.98 |
| WAA / EFE | 60.33 | 63.65 | 63.28 | -3.32 | -2.95 |
| WAA / SPS | 65.53 | 67.26 | 67.22 | -1.73 | -1.69 |
| WAA / SAA | 67.56 | 70.44 | 69.88 | -2.88 | -2.31 |
| WAA / SFE | 64.97 | 68.40 | 67.64 | -3.43 | -2.67 |
| EPS / EFE | 57.93 | 58.25 | 59.85 | -0.33 | -1.93 |
| EPS / SPS | 64.84 | 69.15 | 67.88 | -4.31 | -3.04 |
| EPS / SAA | 64.75 | 67.93 | 66.67 | -3.18 | -1.92 |
| EPS / SFE | 65.30 | 68.99 | 66.77 | -3.69 | -1.47 |
| EFE / SPS | 66.88 | 68.96 | 68.05 | -2.08 | -1.17 |
| EFE / SAA | 65.67 | 68.53 | 68.30 | -2.86 | -2.62 |
| EFE / SFE | 65.11 | 69.00 | 66.73 | -3.89 | -1.62 |
| SPS / SAA | 64.62 | 68.34 | 67.09 | -3.72 | -2.47 |
| SPS / SFE | 64.76 | 68.31 | 66.59 | -3.55 | -1.83 |
| SAA / SFE | 59.66 | 64.18 | 61.78 | -4.52 | -2.12 |

## Notes

- Slim Conformer B is the lowest of the three on all 28 pairs, against both references,
  and no subject prefers it on the window average.
- Its gap to Conformer A widens with window length (-1.27 at 800 ms to -3.86 at 4000 ms);
  its gap to Conformer B stays near -2 at every window.
- Closest calls: HOC / EFE and EPS / EFE, both within 0.3 points of Conformer A.
- The slim model uses about a tenth of Conformer B's parameters and an eighth to a sixth
  of Conformer A's, for 2 to 2.6 points of accuracy.
- The reference runs used A100-PCIE-40GB cards on TACC, these used A100-SXM4-40GB on
  Vast.ai. The numbers are therefore not bit-identical to a TACC run of the same code,
  but the comparison is paired on the same subjects, folds and seeds.
- Per-window and per-pair statistics with confidence intervals: `vs_reference.md` and
  `vs_reference.csv` in this folder.

Generated 2026-09-22 from `results/full/comparison/vs_reference.csv`.
