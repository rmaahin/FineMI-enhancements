# Slim Conformer B vs Conformer A and B (smoke mode, 3 pairs)

Pairs: WAA vs SAA, HOC vs WFE, EPS vs SPS. Reference results: `/workspace/FineMI-enhancements/Conformer_decimated_CWT/results/smoke`.
Paired over 2 subjects. Differences in points, Slim minus reference. 95% CI, paired t-test p and Holm-corrected p within each scope; W = subjects Slim better / reference better.

## Parameters

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms |
|---|---:|---:|---:|---:|
| Slim Conformer B | 25,752 | 25,752 | 25,752 | 25,752 |
| Conformer A | 153,978 | 169,338 | 201,338 | 221,818 |
| Conformer B | 260,034 | 260,034 | 260,034 | 260,034 |

## Mean accuracy, pooled over the selected pairs

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| Slim Conformer B | 50.49 | 52.55 | 50.93 | 53.26 | 51.81 |
| Conformer A | 54.63 | 61.60 | 56.92 | 57.96 | 57.78 |
| Conformer B | 52.15 | 51.55 | 50.21 | 50.69 | 51.15 |

## Pooled over 3 pairs

| Contrast | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| Slim - Conformer A | -4.14 [-52.09, +43.80] (holm=1.000, W 0-2) | -9.05 [-15.82, -2.29] (holm=.374, W 0-2) | -6.00 [-23.94, +11.95] (holm=1.000, W 0-2) | -4.70 [-45.58, +36.18] (holm=1.000, W 0-2) | -5.97 [-25.38, +13.44] (holm=1.000, W 0-2) |
| Slim - Conformer B | -1.67 [-22.84, +19.51] (holm=1.000, W 0-1) | +1.00 [-38.12, +40.11] (holm=1.000, W 1-1) | +0.72 [-27.22, +28.66] (holm=1.000, W 1-1) | +2.57 [-35.37, +40.51] (holm=1.000, W 1-1) | +0.65 [-6.33, +7.64] (holm=1.000, W 2-0) |

## Per pair, window average

| Pair | Contrast | Diff | 95% CI | p | Holm p | W |
|---|---|---:|---:|---:|---:|---:|
| WAA vs SAA | Slim - Conformer A | -9.90 | [-50.93, +31.13] | .201 | 1.000 | 0-2 |
| WAA vs SAA | Slim - Conformer B | -2.26 | [-13.29, +8.77] | .234 | 1.000 | 0-2 |
| HOC vs WFE | Slim - Conformer A | -2.38 | [-7.89, +3.14] | .115 | 1.000 | 0-2 |
| HOC vs WFE | Slim - Conformer B | +2.57 | [-14.20, +19.33] | .302 | 1.000 | 2-0 |
| EPS vs SPS | Slim - Conformer A | -5.64 | [-17.33, +6.05] | .103 | 1.000 | 0-2 |
| EPS vs SPS | Slim - Conformer B | +1.65 | [-5.19, +8.49] | .201 | 1.000 | 2-0 |

## On par? (non-inferiority, margin 1 points, window average)

- Slim vs Conformer A: NOT SHOWN: the interval [-25.38, +13.44] crosses -1; more pairs are needed to decide.
- Slim vs Conformer B: NOT SHOWN: the interval [-6.33, +7.64] crosses -1; more pairs are needed to decide.

Note: smoke-mode numbers only prove the pipeline runs; they say nothing about accuracy.

Runs on a different GPU model (or the CPU) than the reference are not bit-identical to it, but the comparison is still paired on the same subjects, folds and seeds.
