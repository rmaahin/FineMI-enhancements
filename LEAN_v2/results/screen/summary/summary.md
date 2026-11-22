# LEAN_v2, 0.5-3 Hz, 50 Hz (screen mode, 6 pairs, 18 subjects, windows 800 ms, 4000 ms)

Accuracies are means over pairs; comparisons are paired over subjects (each subject's mean over the pairs). Diff = A minus B in percentage points, Holm p over this comparison's windows.

## Mean accuracy

| Model | Params | 800 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|
| R0 LEAN | 25,752 | 60.03 | 55.12 | 57.58 |
| R1 +separable fusion, avg pools | 21,752 | 59.60 | 54.20 | 56.90 |
| R2 +TCN, last-step readout | 14,440 | 60.02 | 53.03 | 56.53 |
| R3 +shared temporal filters | 9,278 | 59.30 | 53.66 | 56.48 |
| R4 +depth attention | 9,286 | 59.50 | 53.49 | 56.49 |
| R5 +ELU/max-norm/dropout, fixed schedule | 9,286 | 66.17 | 64.17 | 65.17 |
| R6 +SupCon | 9,286 | 58.08 | 54.13 | 56.10 |
| R7 +crops (final) | 9,286 | 57.67 | 53.61 | 55.64 |
| LEAN + fixed schedule | 25,752 | 62.54 | 56.62 | 59.58 |
| EEGNet 50 Hz | 1,818 / 2,458 | 63.26 | 63.69 | 63.48 |
| R4 + fixed schedule only | 9,286 | 62.78 | 56.46 | 59.62 |
| R4 + ELU/max-norm/dropout only | 9,286 | 63.81 | 60.18 | 61.99 |
| R5 - separable fusion | 9,862 | 65.82 | 55.67 | 60.75 |
| R5 - TCN (transformer) | 6,662 | 63.91 | 53.76 | 58.84 |
| R5 - last-step readout (GAP+MLP) | 10,918 | 63.68 | 56.87 | 60.27 |
| R5 - shared temporal (and depth att.) | 14,440 | 66.28 | 64.34 | 65.31 |
| R5 - depth attention | 9,278 | 66.50 | 63.81 | 65.16 |
| LEAN + R5 recipe (reg + fixed schedule) | 25,752 | 63.68 | 55.23 | 59.45 |
| EEGNet 50 Hz + fixed schedule | 1,818 / 2,458 | 65.58 | 66.20 | 65.89 |
| LEAN (paper run) | - | 60.81 | 55.32 | 58.07 |
| EEGNet 50 Hz (paper run) | - | 63.31 | 63.89 | 63.60 |

## Ladder steps (each change on top of the rungs before it)

| Comparison | 800 ms | 4000 ms | Avg | Avg 95% CI | Subjects +/- |
|---|---:|---:|---:|---:|---:|
| R1 +separable fusion, avg pools vs R0 | -0.44 (p .512) | -0.92 (p .331) | -0.68 (p .331) | [-1.53, +0.17] | 5-12 |
| R2 +TCN, last-step readout vs R1 | +0.42 (p .926) | -1.17 (p .751) | -0.37 (p .926) | [-1.44, +0.70] | 6-11 |
| R3 +shared temporal filters vs R2 | -0.72 (p .793) | +0.63 (p .793) | -0.04 (p .909) | [-0.86, +0.77] | 9-8 |
| R4 +depth attention vs R3 | +0.20 (p 1.000) | -0.18 (p 1.000) | +0.01 (p 1.000) | [-0.84, +0.87] | 7-11 |
| R5 +ELU/max-norm/dropout, fixed schedule vs R4 | +6.67 (p <.001) | +10.68 (p <.001) | +8.68 (p <.001) | [+6.91, +10.44] | 18-0 |
| R6 +SupCon vs R5 | -8.09 (p <.001) | -10.04 (p <.001) | -9.07 (p <.001) | [-11.04, -7.09] | 0-18 |
| R7 +crops (final) vs R6 | -0.40 (p .922) | -0.52 (p .922) | -0.46 (p .922) | [-1.38, +0.46] | 7-11 |
| R7 vs R0 (all changes) | -2.36 (p .040) | -1.52 (p .080) | -1.94 (p .022) | [-3.27, -0.60] | 4-14 |

## Leave-one-out: R7 minus R7 without the change (positive = the change helps)

| Comparison | 800 ms | 4000 ms | Avg | Avg 95% CI | Subjects +/- |
|---|---:|---:|---:|---:|---:|
| crops | -0.40 (p .922) | -0.52 (p .922) | -0.46 (p .922) | [-1.38, +0.46] | 7-11 |

## Training changes alone, on LEAN (minus R0)

| Comparison | 800 ms | 4000 ms | Avg | Avg 95% CI | Subjects +/- |
|---|---:|---:|---:|---:|---:|
| LEAN + fixed schedule | +2.51 (p <.001) | +1.49 (p .029) | +2.00 (p <.001) | [+1.15, +2.85] | 16-2 |

## Baselines and reproduction checks

| Comparison | 800 ms | 4000 ms | Avg | Avg 95% CI | Subjects +/- |
|---|---:|---:|---:|---:|---:|
| R7 +crops (final) vs EEGNet 50 Hz | -5.59 (p <.001) | -10.08 (p <.001) | -7.83 (p <.001) | [-9.99, -5.68] | 0-18 |
| R0 LEAN vs LEAN (paper run) | -0.78 (p .043) | -0.20 (p .690) | -0.49 (p .159) | [-1.04, +0.06] | 7-10 |
| EEGNet 50 Hz vs EEGNet 50 Hz (paper run) | -0.05 (p .645) | -0.20 (p 1.000) | -0.12 (p 1.000) | [-0.73, +0.48] | 8-10 |

## Decomposing R5 (positive = A better)

| Comparison | 800 ms | 4000 ms | Avg | Avg 95% CI | Subjects +/- |
|---|---:|---:|---:|---:|---:|
| fixed schedule alone (R4 + schedule vs R4) | +3.28 (p <.001) | +2.98 (p <.001) | +3.13 (p <.001) | [+2.31, +3.94] | 18-0 |
| regularisation alone (R4 + reg vs R4) | +4.31 (p <.001) | +6.69 (p <.001) | +5.50 (p <.001) | [+4.02, +6.98] | 18-0 |
| regularisation, given the schedule | +3.40 (p <.001) | +7.71 (p <.001) | +5.55 (p <.001) | [+3.94, +7.16] | 17-1 |
| schedule, given the regularisation | +2.36 (p <.001) | +3.99 (p <.001) | +3.18 (p <.001) | [+2.59, +3.76] | 18-0 |
| R5 minus R5 without separable fusion | +0.35 (p .411) | +8.50 (p <.001) | +4.42 (p <.001) | [+2.88, +5.97] | 17-1 |
| R5 minus R5 without TCN (transformer) | +2.26 (p .005) | +10.41 (p <.001) | +6.33 (p <.001) | [+4.44, +8.23] | 17-1 |
| R5 minus R5 without last-step readout (GAP+MLP) | +2.49 (p <.001) | +7.30 (p <.001) | +4.90 (p <.001) | [+3.70, +6.09] | 17-1 |
| R5 minus R5 without shared temporal (and depth att.) | -0.11 (p 1.000) | -0.17 (p 1.000) | -0.14 (p 1.000) | [-0.89, +0.61] | 9-9 |
| R5 minus R5 without depth attention | -0.33 (p 1.000) | +0.36 (p 1.000) | +0.01 (p 1.000) | [-0.64, +0.66] | 6-11 |
| LEAN + schedule vs LEAN | +2.51 (p <.001) | +1.49 (p .029) | +2.00 (p <.001) | [+1.15, +2.85] | 16-2 |
| LEAN + R5 recipe vs LEAN | +3.65 (p <.001) | +0.11 (p .888) | +1.88 (p .003) | [+0.83, +2.93] | 15-1 |
| R5 vs LEAN, both with the R5 recipe (architecture) | +2.49 (p .002) | +8.94 (p <.001) | +5.72 (p <.001) | [+3.83, +7.60] | 18-0 |
| EEGNet + schedule vs EEGNet | +2.32 (p .003) | +2.51 (p .003) | +2.42 (p <.001) | [+1.43, +3.40] | 17-1 |
| R5 vs EEGNet (both rerun here) | +2.91 (p .003) | +0.48 (p .556) | +1.70 (p .006) | [+0.67, +2.72] | 15-3 |
| R5 vs EEGNet + schedule (same schedule) | +0.59 (p .243) | -2.03 (p .010) | -0.72 (p .152) | [-1.53, +0.08] | 6-11 |

## Every preset minus EEGNet 50 Hz (paper run) (positive = beats EEGNet)

| Comparison | 800 ms | 4000 ms | Avg | Avg 95% CI | Subjects +/- |
|---|---:|---:|---:|---:|---:|
| R0 LEAN | -3.28 (p .001) | -8.77 (p <.001) | -6.02 (p <.001) | [-8.16, -3.89] | 1-17 |
| R1 +separable fusion, avg pools | -3.71 (p .003) | -9.69 (p <.001) | -6.70 (p <.001) | [-8.71, -4.70] | 1-17 |
| R2 +TCN, last-step readout | -3.29 (p .003) | -10.86 (p <.001) | -7.07 (p <.001) | [-8.99, -5.16] | 0-18 |
| R3 +shared temporal filters | -4.01 (p <.001) | -10.23 (p <.001) | -7.12 (p <.001) | [-8.73, -5.51] | 0-18 |
| R4 +depth attention | -3.81 (p <.001) | -10.40 (p <.001) | -7.11 (p <.001) | [-8.98, -5.23] | 0-18 |
| R5 +ELU/max-norm/dropout, fixed schedule | +2.86 (p .003) | +0.28 (p .685) | +1.57 (p .007) | [+0.60, +2.55] | 14-4 |
| R6 +SupCon | -5.23 (p <.001) | -9.77 (p <.001) | -7.50 (p <.001) | [-9.55, -5.45] | 0-18 |
| R7 +crops (final) | -5.64 (p <.001) | -10.28 (p <.001) | -7.96 (p <.001) | [-10.08, -5.84] | 0-18 |
| LEAN + fixed schedule | -0.77 (p .354) | -7.28 (p .002) | -4.02 (p .002) | [-6.11, -1.93] | 3-14 |
| EEGNet 50 Hz | -0.05 (p .645) | -0.20 (p 1.000) | -0.12 (p 1.000) | [-0.73, +0.48] | 8-10 |
| R4 + fixed schedule only | -0.53 (p .540) | -7.43 (p <.001) | -3.98 (p <.001) | [-5.67, -2.29] | 2-16 |
| R4 + ELU/max-norm/dropout only | +0.50 (p .588) | -3.71 (p <.001) | -1.61 (p .025) | [-2.82, -0.39] | 3-14 |
| R5 - separable fusion | +2.51 (p .005) | -8.22 (p <.001) | -2.85 (p .005) | [-4.60, -1.11] | 5-13 |
| R5 - TCN (transformer) | +0.60 (p .489) | -10.13 (p <.001) | -4.76 (p <.001) | [-6.89, -2.64] | 3-15 |
| R5 - last-step readout (GAP+MLP) | +0.37 (p .612) | -7.03 (p <.001) | -3.33 (p <.001) | [-4.83, -1.82] | 2-16 |
| R5 - shared temporal (and depth att.) | +2.97 (p .010) | +0.45 (p .519) | +1.71 (p .010) | [+0.63, +2.79] | 15-3 |
| R5 - depth attention | +3.19 (p .002) | -0.08 (p .912) | +1.56 (p .028) | [+0.36, +2.76] | 12-6 |
| LEAN + R5 recipe (reg + fixed schedule) | +0.37 (p .670) | -8.66 (p <.001) | -4.15 (p .001) | [-6.28, -2.01] | 2-16 |
| EEGNet 50 Hz + fixed schedule | +2.28 (p .002) | +2.31 (p .002) | +2.29 (p <.001) | [+1.29, +3.30] | 15-3 |

## By joint group (window average)

| Group | Pairs | R0 LEAN | R1 +separable fusion, avg pools | R2 +TCN, last-step readout | R3 +shared temporal filters | R4 +depth attention | R5 +ELU/max-norm/dropout, fixed schedule | R6 +SupCon | R7 +crops (final) | LEAN + fixed schedule | EEGNet 50 Hz | R4 + fixed schedule only | R4 + ELU/max-norm/dropout only | R5 - separable fusion | R5 - TCN (transformer) | R5 - last-step readout (GAP+MLP) | R5 - shared temporal (and depth att.) | R5 - depth attention | LEAN + R5 recipe (reg + fixed schedule) | EEGNet 50 Hz + fixed schedule | LEAN (paper run) | EEGNet 50 Hz (paper run) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Distant | 2 | 59.23 | 58.72 | 58.56 | 58.42 | 57.37 | 68.33 | 56.80 | 56.01 | 62.18 | 66.39 | 61.28 | 64.80 | 63.53 | 60.68 | 62.22 | 68.14 | 67.91 | 62.03 | 69.17 | 59.85 | 66.59 |
| Adjacent | 2 | 57.36 | 55.35 | 55.35 | 55.74 | 56.37 | 64.16 | 55.50 | 55.68 | 57.84 | 62.83 | 58.51 | 60.83 | 59.12 | 58.11 | 60.30 | 63.73 | 64.35 | 58.05 | 64.44 | 57.50 | 63.08 |
| Same-joint | 2 | 56.14 | 56.63 | 55.67 | 55.29 | 55.74 | 63.03 | 56.01 | 55.23 | 58.72 | 61.20 | 59.08 | 60.36 | 59.59 | 57.71 | 58.30 | 64.06 | 63.21 | 58.28 | 64.07 | 56.86 | 61.13 |
| Overall | 6 | 57.58 | 56.90 | 56.53 | 56.48 | 56.49 | 65.17 | 56.10 | 55.64 | 59.58 | 63.48 | 59.62 | 61.99 | 60.75 | 58.84 | 60.27 | 65.31 | 65.16 | 59.45 | 65.89 | 58.07 | 63.60 |

## Per pair (window average)

| Pair | Group | R0 LEAN | R1 +separable fusion, avg pools | R2 +TCN, last-step readout | R3 +shared temporal filters | R4 +depth attention | R5 +ELU/max-norm/dropout, fixed schedule | R6 +SupCon | R7 +crops (final) | LEAN + fixed schedule | EEGNet 50 Hz | R4 + fixed schedule only | R4 + ELU/max-norm/dropout only | R5 - separable fusion | R5 - TCN (transformer) | R5 - last-step readout (GAP+MLP) | R5 - shared temporal (and depth att.) | R5 - depth attention | LEAN + R5 recipe (reg + fixed schedule) | EEGNet 50 Hz + fixed schedule | LEAN (paper run) | EEGNet 50 Hz (paper run) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HOC/SPS | Distant | 60.45 | 59.57 | 58.42 | 59.46 | 57.84 | 68.19 | 58.03 | 57.54 | 62.56 | 68.05 | 62.11 | 65.03 | 63.21 | 60.17 | 63.92 | 69.20 | 69.00 | 62.04 | 70.64 | 60.64 | 68.26 |
| WAA/SFE | Distant | 58.00 | 57.87 | 58.69 | 57.38 | 56.89 | 68.47 | 55.58 | 54.49 | 61.81 | 64.74 | 60.44 | 64.56 | 63.86 | 61.20 | 60.52 | 67.08 | 66.82 | 62.02 | 67.69 | 59.05 | 64.92 |
| HOC/WFE | Adjacent | 54.75 | 51.02 | 53.82 | 53.24 | 54.13 | 60.95 | 53.98 | 53.70 | 54.03 | 58.79 | 56.14 | 57.67 | 56.82 | 54.88 | 56.52 | 59.29 | 60.74 | 54.89 | 60.43 | 54.44 | 59.92 |
| EPS/SAA | Adjacent | 59.98 | 59.67 | 56.87 | 58.24 | 58.62 | 67.37 | 57.01 | 57.66 | 61.64 | 66.87 | 60.87 | 63.99 | 61.42 | 61.35 | 64.07 | 68.17 | 67.96 | 61.22 | 68.45 | 60.56 | 66.24 |
| EPS/EFE | Same-joint | 54.52 | 55.20 | 53.60 | 54.12 | 54.33 | 59.95 | 54.46 | 53.45 | 57.02 | 57.61 | 56.79 | 56.86 | 57.65 | 55.10 | 57.22 | 61.13 | 60.42 | 57.10 | 61.06 | 55.91 | 57.72 |
| SPS/SAA | Same-joint | 57.76 | 58.06 | 57.75 | 56.45 | 57.14 | 66.10 | 57.55 | 57.01 | 60.42 | 64.80 | 61.36 | 63.85 | 61.53 | 60.32 | 59.39 | 66.99 | 66.00 | 59.46 | 67.08 | 57.80 | 64.54 |

