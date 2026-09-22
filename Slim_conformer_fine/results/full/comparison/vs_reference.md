# Slim Conformer B vs Conformer A and B (full mode, 3 pairs)

Pairs: WAA vs SAA, HOC vs WFE, EPS vs SPS. Reference results: `C:\Users\rmaah\Documents\Summer 2026\Fine MI\FineMI-enhancements\Conformer_decimated_CWT\results\full`.
Paired over 18 subjects. Differences in points, Slim minus reference. 95% CI, paired t-test p and Holm-corrected p within each scope; W = subjects Slim better / reference better.

## Parameters

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms |
|---|---:|---:|---:|---:|
| Slim Conformer B | 25,752 | 25,752 | 25,752 | 25,752 |
| Conformer A | 153,978 | 169,338 | 201,338 | 221,818 |
| Conformer B | 260,034 | 260,034 | 260,034 | 260,034 |

## Mean accuracy, pooled over the selected pairs

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| Slim Conformer B | 62.35 | 66.34 | 64.16 | 62.57 | 63.86 |
| Conformer A | 63.09 | 67.63 | 68.84 | 68.04 | 66.90 |
| Conformer B | 65.16 | 67.88 | 65.16 | 65.70 | 65.98 |

## Pooled over 3 pairs

| Contrast | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| Slim - Conformer A | -0.74 [-2.16, +0.69] (holm=.580, W 9-9) | -1.29 [-3.15, +0.56] (holm=.478, W 3-14) | -4.68 [-7.24, -2.11] (holm=.008, W 3-15) | -5.47 [-7.99, -2.94] (holm=.002, W 2-16) | -3.04 [-4.16, -1.93] (holm=<.001, W 2-15) |
| Slim - Conformer B | -2.81 [-3.96, -1.67] (holm=<.001, W 3-15) | -1.54 [-3.28, +0.20] (holm=.314, W 3-14) | -0.99 [-3.14, +1.15] (holm=.580, W 6-10) | -3.13 [-4.86, -1.40] (holm=.008, W 3-13) | -2.12 [-3.02, -1.22] (holm=<.001, W 2-16) |

## Per pair, window average

| Pair | Contrast | Diff | 95% CI | p | Holm p | W |
|---|---|---:|---:|---:|---:|---:|
| WAA vs SAA | Slim - Conformer A | -2.38 | [-4.26, -0.50] | .016 | .341 | 6-11 |
| WAA vs SAA | Slim - Conformer B | -1.81 | [-3.22, -0.41] | .014 | .329 | 4-13 |
| HOC vs WFE | Slim - Conformer A | -2.57 | [-5.14, -0.01] | .049 | .889 | 6-11 |
| HOC vs WFE | Slim - Conformer B | -1.64 | [-3.35, +0.07] | .059 | 1.000 | 7-11 |
| EPS vs SPS | Slim - Conformer A | -4.18 | [-6.18, -2.18] | <.001 | .011 | 2-16 |
| EPS vs SPS | Slim - Conformer B | -2.91 | [-4.67, -1.15] | .003 | .076 | 2-16 |

## On par? (non-inferiority, margin 1 points, window average)

- Slim vs Conformer A: WORSE: the whole interval is below -1 (-1.93 upper end).
- Slim vs Conformer B: WORSE: the whole interval is below -1 (-1.22 upper end).

Runs on a different GPU model (or the CPU) than the reference are not bit-identical to it, but the comparison is still paired on the same subjects, folds and seeds.
