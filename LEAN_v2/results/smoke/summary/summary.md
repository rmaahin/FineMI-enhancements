# LEAN_v2, 0.5-3 Hz, 50 Hz (smoke mode, 3 pairs, 2 subjects, windows 800 ms, 1500 ms, 3000 ms, 4000 ms)

Accuracies are means over pairs; comparisons are paired over subjects (each subject's mean over the pairs). Diff = A minus B in percentage points, Holm p over this comparison's windows.

## Mean accuracy

| Model | Params | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|---:|
| R0 LEAN | 25,752 | 51.88 | 49.56 | 46.71 | 50.25 | 49.60 |
| R1 +separable fusion, avg pools | 21,752 | 51.76 | 49.91 | 51.18 | 52.87 | 51.43 |
| R2 +TCN, last-step readout | 14,440 | 49.26 | 48.89 | 49.63 | 52.82 | 50.15 |
| R3 +shared temporal filters | 9,278 | 50.19 | 50.00 | 51.11 | 50.00 | 50.32 |
| R4 +depth attention | 9,286 | 49.77 | 50.19 | 51.64 | 49.63 | 50.31 |
| R5 +ELU/max-norm/dropout, fixed schedule | 9,286 | 53.89 | 50.60 | 48.19 | 49.81 | 50.62 |
| R6 +SupCon | 9,286 | 52.22 | 50.79 | 45.95 | 51.30 | 50.06 |
| R7 +crops (final) | 9,286 | 51.25 | 48.75 | 48.10 | 49.79 | 49.47 |
| R7 - separable fusion | 9,862 | 51.16 | 49.19 | 49.03 | 54.00 | 50.84 |
| R7 - TCN (transformer) | 6,614 | 48.06 | 50.51 | 51.41 | 46.20 | 49.05 |
| R7 - last-step readout (GAP+MLP) | 10,918 | 50.37 | 49.61 | 50.00 | 47.78 | 49.44 |
| R7 - shared temporal (and depth att.) | 14,440 | 49.47 | 50.56 | 50.56 | 50.58 | 50.29 |
| R7 - depth attention | 9,278 | 49.35 | 50.16 | 49.58 | 51.25 | 50.09 |
| R7 - ELU/max-norm/dropout | 9,286 | 49.77 | 50.19 | 50.19 | 50.74 | 50.22 |
| R7 - fixed schedule (best-val) | 9,286 | 52.08 | 48.77 | 47.89 | 50.02 | 49.69 |
| R7 - SupCon | 9,286 | 51.88 | 50.32 | 48.61 | 50.37 | 50.30 |
| LEAN + fixed schedule | 25,752 | 50.86 | 50.14 | 48.26 | 50.35 | 49.90 |
| LEAN + SupCon | 25,752 | 48.63 | 50.05 | 49.21 | 52.04 | 49.98 |
| LEAN + R7 recipe | 25,560 | 48.89 | 48.82 | 46.20 | 48.33 | 48.06 |
| EEGNet 50 Hz | 1,818 / 1,946 / 2,234 / 2,458 | 54.88 | 50.95 | 57.50 | 59.54 | 55.72 |
| EEGNet 50 Hz + R7 recipe | 1,786 / 1,914 / 2,170 / 2,362 | 51.50 | 51.53 | 57.50 | 53.10 | 53.41 |

## Ladder steps (each change on top of the rungs before it)

| Comparison | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg | Avg 95% CI | Subjects +/- |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 +separable fusion, avg pools vs R0 | -0.12 (p 1.000) | +0.35 (p 1.000) | +4.47 (p .507) | +2.62 (p 1.000) | +1.83 (p .780) | [-5.52, +9.18] | 2-0 |
| R2 +TCN, last-step readout vs R1 | -2.50 (p .819) | -1.02 (p 1.000) | -1.55 (p .610) | -0.05 (p 1.000) | -1.28 (p 1.000) | [-10.91, +8.35] | 0-2 |
| R3 +shared temporal filters vs R2 | +0.93 (p 1.000) | +1.11 (p 1.000) | +1.48 (p 1.000) | -2.82 (p 1.000) | +0.17 (p 1.000) | [-15.27, +15.62] | 1-1 |
| R4 +depth attention vs R3 | -0.42 (p 1.000) | +0.19 (p 1.000) | +0.53 (p 1.000) | -0.37 (p 1.000) | -0.02 (p 1.000) | [-6.86, +6.82] | 1-1 |
| R5 +ELU/max-norm/dropout, fixed schedule vs R4 | +4.12 (p 1.000) | +0.42 (p 1.000) | -3.45 (p 1.000) | +0.19 (p 1.000) | +0.32 (p 1.000) | [-4.90, +5.54] | 1-1 |
| R6 +SupCon vs R5 | -1.67 (p 1.000) | +0.19 (p 1.000) | -2.25 (p 1.000) | +1.48 (p 1.000) | -0.56 (p 1.000) | [-15.93, +14.81] | 1-1 |
| R7 +crops (final) vs R6 | -0.97 (p .361) | -2.04 (p 1.000) | +2.15 (p .103) | -1.50 (p 1.000) | -0.59 (p 1.000) | [-10.74, +9.56] | 1-1 |
| R7 vs R0 (all changes) | -0.62 (p 1.000) | -0.81 (p 1.000) | +1.39 (p 1.000) | -0.46 (p 1.000) | -0.13 (p 1.000) | [-6.45, +6.20] | 1-1 |

## Leave-one-out: R7 minus R7 without the change (positive = the change helps)

| Comparison | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg | Avg 95% CI | Subjects +/- |
|---|---:|---:|---:|---:|---:|---:|---:|
| separable fusion | +0.09 (p 1.000) | -0.44 (p 1.000) | -0.93 (p 1.000) | -4.21 (p .935) | -1.37 (p .040) | [-1.59, -1.15] | 0-2 |
| TCN (transformer) | +3.19 (p 1.000) | -1.76 (p 1.000) | -3.31 (p .377) | +3.59 (p 1.000) | +0.43 (p 1.000) | [-39.13, +39.99] | 1-1 |
| last-step readout (GAP+MLP) | +0.88 (p 1.000) | -0.86 (p 1.000) | -1.90 (p 1.000) | +2.01 (p 1.000) | +0.03 (p 1.000) | [-3.05, +3.12] | 1-1 |
| shared temporal (and depth att.) | +1.78 (p .426) | -1.81 (p .776) | -2.45 (p .776) | -0.79 (p .187) | -0.82 (p .663) | [-4.57, +2.93] | 0-2 |
| depth attention | +1.90 (p 1.000) | -1.41 (p 1.000) | -1.48 (p 1.000) | -1.46 (p 1.000) | -0.61 (p 1.000) | [-5.76, +4.53] | 0-2 |
| ELU/max-norm/dropout | +1.48 (p 1.000) | -1.44 (p 1.000) | -2.08 (p 1.000) | -0.95 (p 1.000) | -0.75 (p 1.000) | [-8.91, +7.42] | 0-2 |
| fixed schedule (best-val) | -0.83 (p 1.000) | -0.02 (p 1.000) | +0.21 (p 1.000) | -0.23 (p 1.000) | -0.22 (p 1.000) | [-5.37, +4.93] | 1-1 |
| SupCon | -0.62 (p 1.000) | -1.57 (p .187) | -0.51 (p 1.000) | -0.58 (p .695) | -0.82 (p 1.000) | [-8.91, +7.27] | 0-2 |
| crops | -0.97 (p .361) | -2.04 (p 1.000) | +2.15 (p .103) | -1.50 (p 1.000) | -0.59 (p 1.000) | [-10.74, +9.56] | 1-1 |

## Training changes alone, on LEAN (minus R0)

| Comparison | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg | Avg 95% CI | Subjects +/- |
|---|---:|---:|---:|---:|---:|---:|---:|
| LEAN + fixed schedule | -1.02 (p 1.000) | +0.58 (p .869) | +1.55 (p 1.000) | +0.09 (p 1.000) | +0.30 (p 1.000) | [-11.76, +12.36] | 1-1 |
| LEAN + SupCon | -3.24 (p .819) | +0.49 (p .826) | +2.50 (p .819) | +1.78 (p .532) | +0.38 (p .819) | [-1.82, +2.59] | 2-0 |
| LEAN + R7 recipe | -2.99 (p 1.000) | -0.74 (p 1.000) | -0.51 (p 1.000) | -1.92 (p .856) | -1.54 (p .683) | [-5.80, +2.73] | 0-2 |

## Baselines and reproduction checks

| Comparison | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg | Avg 95% CI | Subjects +/- |
|---|---:|---:|---:|---:|---:|---:|---:|
| EEGNet 50 Hz + R7 recipe vs EEGNet 50 Hz | -3.38 (p .291) | +0.58 (p .291) | +0.00 (p -) | -6.44 (p .178) | -2.31 (p .096) | [-3.41, -1.21] | 0-2 |
| R7 +crops (final) vs EEGNet 50 Hz + R7 recipe | -0.25 (p 1.000) | -2.78 (p .628) | -9.40 (p 1.000) | -3.31 (p 1.000) | -3.94 (p 1.000) | [-38.64, +30.77] | 0-2 |
| R7 +crops (final) vs EEGNet 50 Hz | -3.63 (p .810) | -2.20 (p .803) | -9.40 (p .810) | -9.75 (p .749) | -6.24 (p .810) | [-42.05, +29.57] | 0-2 |

## By joint group (window average)

| Group | Pairs | R0 LEAN | R1 +separable fusion, avg pools | R2 +TCN, last-step readout | R3 +shared temporal filters | R4 +depth attention | R5 +ELU/max-norm/dropout, fixed schedule | R6 +SupCon | R7 +crops (final) | R7 - separable fusion | R7 - TCN (transformer) | R7 - last-step readout (GAP+MLP) | R7 - shared temporal (and depth att.) | R7 - depth attention | R7 - ELU/max-norm/dropout | R7 - fixed schedule (best-val) | R7 - SupCon | LEAN + fixed schedule | LEAN + SupCon | LEAN + R7 recipe | EEGNet 50 Hz | EEGNet 50 Hz + R7 recipe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Distant | 1 | 47.53 | 53.28 | 49.60 | 50.42 | 51.28 | 51.22 | 50.85 | 48.91 | 50.36 | 50.03 | 50.17 | 50.87 | 50.26 | 50.38 | 49.62 | 51.13 | 49.60 | 49.24 | 48.28 | 54.38 | 53.94 |
| Adjacent | 2 | 50.63 | 50.50 | 50.43 | 50.28 | 49.82 | 50.33 | 49.67 | 49.76 | 51.09 | 48.55 | 49.07 | 50.00 | 50.00 | 50.14 | 49.73 | 49.88 | 50.05 | 50.36 | 47.95 | 56.39 | 53.14 |
| Overall | 3 | 49.60 | 51.43 | 50.15 | 50.32 | 50.31 | 50.62 | 50.06 | 49.47 | 50.84 | 49.05 | 49.44 | 50.29 | 50.09 | 50.22 | 49.69 | 50.30 | 49.90 | 49.98 | 48.06 | 55.72 | 53.41 |

## Per pair (window average)

| Pair | Group | R0 LEAN | R1 +separable fusion, avg pools | R2 +TCN, last-step readout | R3 +shared temporal filters | R4 +depth attention | R5 +ELU/max-norm/dropout, fixed schedule | R6 +SupCon | R7 +crops (final) | R7 - separable fusion | R7 - TCN (transformer) | R7 - last-step readout (GAP+MLP) | R7 - shared temporal (and depth att.) | R7 - depth attention | R7 - ELU/max-norm/dropout | R7 - fixed schedule (best-val) | R7 - SupCon | LEAN + fixed schedule | LEAN + SupCon | LEAN + R7 recipe | EEGNet 50 Hz | EEGNet 50 Hz + R7 recipe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WAA/SAA | Distant | 47.53 | 53.28 | 49.60 | 50.42 | 51.28 | 51.22 | 50.85 | 48.91 | 50.36 | 50.03 | 50.17 | 50.87 | 50.26 | 50.38 | 49.62 | 51.13 | 49.60 | 49.24 | 48.28 | 54.38 | 53.94 |
| HOC/WFE | Adjacent | 49.90 | 51.09 | 50.33 | 50.14 | 50.75 | 50.00 | 48.30 | 49.25 | 53.28 | 48.02 | 49.22 | 50.56 | 49.79 | 50.68 | 49.17 | 49.43 | 49.34 | 50.56 | 47.53 | 55.90 | 52.34 |
| EPS/SPS | Adjacent | 51.37 | 49.91 | 50.52 | 50.42 | 48.89 | 50.66 | 51.04 | 50.26 | 48.89 | 49.08 | 48.92 | 49.44 | 50.21 | 49.60 | 50.30 | 50.33 | 50.76 | 50.16 | 48.37 | 56.88 | 53.94 |

Note: smoke-mode numbers only prove the pipeline runs; they say nothing about accuracy.

