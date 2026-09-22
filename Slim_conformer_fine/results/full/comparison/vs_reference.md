# Slim Conformer B vs Conformer A and B (full mode, 28 pairs)

Pairs: HOC vs WFE, HOC vs WAA, HOC vs EPS, HOC vs EFE, HOC vs SPS, HOC vs SAA, HOC vs SFE, WFE vs WAA, WFE vs EPS, WFE vs EFE, WFE vs SPS, WFE vs SAA, WFE vs SFE, WAA vs EPS, WAA vs EFE, WAA vs SPS, WAA vs SAA, WAA vs SFE, EPS vs EFE, EPS vs SPS, EPS vs SAA, EPS vs SFE, EFE vs SPS, EFE vs SAA, EFE vs SFE, SPS vs SAA, SPS vs SFE, SAA vs SFE. Reference results: `/workspace/FineMI-enhancements/Conformer_decimated_CWT/results/full`.
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
| Slim Conformer B | 61.78 | 64.80 | 64.40 | 63.94 | 63.73 |
| Conformer A | 63.05 | 66.61 | 67.96 | 67.80 | 66.35 |
| Conformer B | 63.83 | 66.68 | 66.46 | 66.11 | 65.77 |

## Pooled over 28 pairs

| Contrast | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| Slim - Conformer A | -1.27 [-1.78, -0.76] (holm=<.001, W 2-16) | -1.81 [-2.70, -0.92] (holm=<.001, W 2-15) | -3.55 [-4.61, -2.50] (holm=<.001, W 1-17) | -3.86 [-4.87, -2.84] (holm=<.001, W 0-18) | -2.62 [-3.18, -2.07] (holm=<.001, W 0-18) |
| Slim - Conformer B | -2.06 [-2.64, -1.47] (holm=<.001, W 1-17) | -1.87 [-2.66, -1.08] (holm=<.001, W 1-17) | -2.06 [-2.57, -1.55] (holm=<.001, W 0-18) | -2.16 [-2.74, -1.59] (holm=<.001, W 1-17) | -2.04 [-2.40, -1.68] (holm=<.001, W 0-18) |

## Per pair, window average

| Pair | Contrast | Diff | 95% CI | p | Holm p | W |
|---|---|---:|---:|---:|---:|---:|
| HOC vs WFE | Slim - Conformer A | -2.24 | [-3.84, -0.63] | .009 | 1.000 | 4-13 |
| HOC vs WFE | Slim - Conformer B | -1.30 | [-2.75, +0.14] | .074 | 1.000 | 6-10 |
| HOC vs WAA | Slim - Conformer A | -3.06 | [-4.67, -1.45] | <.001 | .241 | 3-15 |
| HOC vs WAA | Slim - Conformer B | -4.04 | [-5.69, -2.39] | <.001 | .021 | 2-16 |
| HOC vs EPS | Slim - Conformer A | -2.37 | [-3.79, -0.94] | .003 | .666 | 5-12 |
| HOC vs EPS | Slim - Conformer B | -2.27 | [-3.65, -0.89] | .003 | .733 | 3-14 |
| HOC vs EFE | Slim - Conformer A | -0.23 | [-1.69, +1.22] | .740 | 1.000 | 7-8 |
| HOC vs EFE | Slim - Conformer B | -2.32 | [-3.71, -0.93] | .003 | .662 | 4-14 |
| HOC vs SPS | Slim - Conformer A | -1.38 | [-3.06, +0.30] | .102 | 1.000 | 8-9 |
| HOC vs SPS | Slim - Conformer B | -1.08 | [-2.46, +0.29] | .116 | 1.000 | 6-12 |
| HOC vs SAA | Slim - Conformer A | -2.11 | [-3.55, -0.67] | .007 | 1.000 | 4-13 |
| HOC vs SAA | Slim - Conformer B | -1.86 | [-3.21, -0.50] | .010 | 1.000 | 7-11 |
| HOC vs SFE | Slim - Conformer A | -2.04 | [-3.96, -0.11] | .040 | 1.000 | 6-12 |
| HOC vs SFE | Slim - Conformer B | -2.78 | [-4.48, -1.09] | .003 | .724 | 3-15 |
| WFE vs WAA | Slim - Conformer A | -0.98 | [-3.87, +1.91] | .484 | 1.000 | 9-9 |
| WFE vs WAA | Slim - Conformer B | -1.49 | [-3.03, +0.06] | .059 | 1.000 | 5-11 |
| WFE vs EPS | Slim - Conformer A | -2.20 | [-3.84, -0.57] | .011 | 1.000 | 2-14 |
| WFE vs EPS | Slim - Conformer B | -1.56 | [-3.48, +0.36] | .105 | 1.000 | 4-12 |
| WFE vs EFE | Slim - Conformer A | -2.07 | [-3.80, -0.35] | .021 | 1.000 | 4-14 |
| WFE vs EFE | Slim - Conformer B | -1.60 | [-3.07, -0.13] | .035 | 1.000 | 7-11 |
| WFE vs SPS | Slim - Conformer A | -1.29 | [-3.17, +0.59] | .166 | 1.000 | 9-9 |
| WFE vs SPS | Slim - Conformer B | -1.02 | [-2.32, +0.28] | .115 | 1.000 | 7-10 |
| WFE vs SAA | Slim - Conformer A | -3.22 | [-4.46, -1.97] | <.001 | .012 | 1-15 |
| WFE vs SAA | Slim - Conformer B | -1.46 | [-2.70, -0.21] | .025 | 1.000 | 6-12 |
| WFE vs SFE | Slim - Conformer A | -3.58 | [-5.37, -1.80] | <.001 | .151 | 3-15 |
| WFE vs SFE | Slim - Conformer B | -2.51 | [-4.31, -0.71] | .009 | 1.000 | 4-14 |
| WAA vs EPS | Slim - Conformer A | -3.15 | [-5.11, -1.20] | .003 | .833 | 3-14 |
| WAA vs EPS | Slim - Conformer B | -1.98 | [-3.82, -0.14] | .036 | 1.000 | 6-12 |
| WAA vs EFE | Slim - Conformer A | -3.32 | [-4.96, -1.69] | <.001 | .137 | 3-15 |
| WAA vs EFE | Slim - Conformer B | -2.95 | [-4.15, -1.75] | <.001 | .021 | 1-16 |
| WAA vs SPS | Slim - Conformer A | -1.73 | [-3.24, -0.22] | .027 | 1.000 | 4-14 |
| WAA vs SPS | Slim - Conformer B | -1.69 | [-2.80, -0.58] | .005 | 1.000 | 3-12 |
| WAA vs SAA | Slim - Conformer A | -2.88 | [-4.81, -0.95] | .006 | 1.000 | 4-14 |
| WAA vs SAA | Slim - Conformer B | -2.31 | [-3.52, -1.10] | <.001 | .231 | 2-14 |
| WAA vs SFE | Slim - Conformer A | -3.43 | [-5.14, -1.72] | <.001 | .153 | 3-15 |
| WAA vs SFE | Slim - Conformer B | -2.67 | [-3.61, -1.73] | <.001 | .004 | 1-17 |
| EPS vs EFE | Slim - Conformer A | -0.33 | [-2.41, +1.76] | .744 | 1.000 | 8-10 |
| EPS vs EFE | Slim - Conformer B | -1.93 | [-3.07, -0.78] | .002 | .621 | 2-15 |
| EPS vs SPS | Slim - Conformer A | -4.31 | [-6.40, -2.22] | <.001 | .117 | 3-15 |
| EPS vs SPS | Slim - Conformer B | -3.04 | [-4.62, -1.46] | <.001 | .218 | 2-16 |
| EPS vs SAA | Slim - Conformer A | -3.18 | [-5.38, -0.98] | .007 | 1.000 | 4-14 |
| EPS vs SAA | Slim - Conformer B | -1.92 | [-3.86, +0.02] | .052 | 1.000 | 4-14 |
| EPS vs SFE | Slim - Conformer A | -3.69 | [-5.74, -1.65] | .001 | .365 | 5-13 |
| EPS vs SFE | Slim - Conformer B | -1.47 | [-2.68, -0.25] | .021 | 1.000 | 4-13 |
| EFE vs SPS | Slim - Conformer A | -2.08 | [-4.00, -0.17] | .035 | 1.000 | 3-11 |
| EFE vs SPS | Slim - Conformer B | -1.17 | [-2.42, +0.09] | .066 | 1.000 | 5-11 |
| EFE vs SAA | Slim - Conformer A | -2.86 | [-5.12, -0.60] | .016 | 1.000 | 4-14 |
| EFE vs SAA | Slim - Conformer B | -2.62 | [-4.35, -0.89] | .005 | 1.000 | 2-16 |
| EFE vs SFE | Slim - Conformer A | -3.89 | [-6.02, -1.76] | .001 | .328 | 2-16 |
| EFE vs SFE | Slim - Conformer B | -1.62 | [-2.80, -0.44] | .010 | 1.000 | 5-13 |
| SPS vs SAA | Slim - Conformer A | -3.72 | [-5.82, -1.63] | .002 | .412 | 3-15 |
| SPS vs SAA | Slim - Conformer B | -2.47 | [-3.99, -0.96] | .003 | .756 | 2-15 |
| SPS vs SFE | Slim - Conformer A | -3.55 | [-5.67, -1.43] | .003 | .640 | 4-13 |
| SPS vs SFE | Slim - Conformer B | -1.83 | [-3.68, +0.02] | .052 | 1.000 | 6-11 |
| SAA vs SFE | Slim - Conformer A | -4.52 | [-6.45, -2.58] | <.001 | .036 | 2-16 |
| SAA vs SFE | Slim - Conformer B | -2.12 | [-4.05, -0.19] | .033 | 1.000 | 4-14 |

## On par? (non-inferiority, margin 1 points, window average)

- Slim vs Conformer A: WORSE: the whole interval is below -1 (-2.07 upper end).
- Slim vs Conformer B: WORSE: the whole interval is below -1 (-1.68 upper end).

Runs on a different GPU model (or the CPU) than the reference are not bit-identical to it, but the comparison is still paired on the same subjects, folds and seeds.
