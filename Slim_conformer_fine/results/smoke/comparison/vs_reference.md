# Slim Conformer B vs Conformer A and B (smoke mode, 3 pairs)

Pairs: WAA vs SAA, HOC vs WFE, EPS vs SPS. Reference results: `C:\Users\rmaah\Documents\Summer 2026\Fine MI\FineMI-enhancements\Conformer_decimated_CWT\results\smoke`.
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
| Slim Conformer B | 50.67 | 51.00 | 53.08 | 47.85 | 50.65 |
| Conformer A | 54.63 | 61.60 | 56.92 | 57.96 | 57.78 |
| Conformer B | 52.15 | 51.55 | 50.21 | 50.69 | 51.15 |

## Pooled over 3 pairs

| Contrast | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| Slim - Conformer A | -3.96 [-54.25, +46.34] (holm=1.000, W 0-1) | -10.60 [-13.54, -7.66] (holm=.139, W 0-2) | -3.84 [-29.14, +21.45] (holm=1.000, W 0-2) | -10.12 [-77.47, +57.24] (holm=1.000, W 0-2) | -7.13 [-43.60, +29.34] (holm=1.000, W 0-2) |
| Slim - Conformer B | -1.48 [-20.31, +17.34] (holm=1.000, W 0-1) | -0.56 [-35.85, +34.74] (holm=1.000, W 1-1) | +2.87 [-12.42, +18.16] (holm=1.000, W 2-0) | -2.85 [-67.26, +61.57] (holm=1.000, W 1-1) | -0.50 [-24.55, +23.54] (holm=1.000, W 1-1) |

## Per pair, window average

| Pair | Contrast | Diff | 95% CI | p | Holm p | W |
|---|---|---:|---:|---:|---:|---:|
| WAA vs SAA | Slim - Conformer A | -9.69 | [-69.25, +49.87] | .287 | 1.000 | 0-2 |
| WAA vs SAA | Slim - Conformer B | -2.05 | [-31.61, +27.51] | .540 | 1.000 | 1-1 |
| HOC vs WFE | Slim - Conformer A | -4.48 | [-19.04, +10.08] | .159 | 1.000 | 0-2 |
| HOC vs WFE | Slim - Conformer B | +0.47 | [-25.34, +26.28] | .856 | 1.000 | 1-1 |
| EPS vs SPS | Slim - Conformer A | -7.22 | [-42.52, +28.07] | .234 | 1.000 | 0-2 |
| EPS vs SPS | Slim - Conformer B | +0.07 | [-16.70, +16.83] | .967 | 1.000 | 1-1 |

## On par? (non-inferiority, margin 1 points, window average)

- Slim vs Conformer A: NOT SHOWN: the interval [-43.60, +29.34] crosses -1; more pairs are needed to decide.
- Slim vs Conformer B: NOT SHOWN: the interval [-24.55, +23.54] crosses -1; more pairs are needed to decide.

Note: smoke-mode numbers only prove the pipeline runs; they say nothing about accuracy.

Runs on a different GPU model (or the CPU) than the reference are not bit-identical to it, but the comparison is still paired on the same subjects, folds and seeds.
