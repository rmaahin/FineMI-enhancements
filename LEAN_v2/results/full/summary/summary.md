# LEAN_v2, 0.5-3 Hz, 50 Hz (full mode, 28 pairs, 18 subjects, windows 800 ms, 1500 ms, 3000 ms, 4000 ms)

Accuracies are means over pairs; comparisons are paired over subjects (each subject's mean over the pairs). Diff = A minus B in percentage points, Holm p over this comparison's windows.

## Mean accuracy

| Model | Params | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|---:|
| EEGNet 50 Hz + fixed schedule | 1,818 / 1,946 / 2,234 / 2,458 | 64.36 | 65.11 | 66.66 | 65.77 | 65.48 |
| LEAN v2 (final) | 9,278 | 65.25 | 66.47 | 67.16 | 64.60 | 65.87 |
| LEAN (paper run) | - | 60.06 | 58.29 | 55.67 | 55.17 | 57.30 |
| EEGNet 50 Hz (paper run) | - | 62.71 | 63.26 | 64.28 | 63.42 | 63.41 |

## Final model (positive = LEAN v2 better)

| Comparison | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg | Avg 95% CI | Subjects +/- |
|---|---:|---:|---:|---:|---:|---:|---:|
| LEAN v2 (final) vs EEGNet 50 Hz + fixed schedule | +0.89 (p .008) | +1.35 (p .010) | +0.50 (p .191) | -1.17 (p .055) | +0.39 (p .083) | [+0.02, +0.77] | 12-6 |
| LEAN v2 (final) vs EEGNet 50 Hz (paper run) | +2.55 (p <.001) | +3.21 (p <.001) | +2.89 (p <.001) | +1.18 (p .005) | +2.46 (p <.001) | [+1.91, +3.01] | 18-0 |
| LEAN v2 (final) vs LEAN (paper run) | +5.19 (p <.001) | +8.17 (p <.001) | +11.49 (p <.001) | +9.43 (p <.001) | +8.57 (p <.001) | [+6.55, +10.60] | 18-0 |

## Every preset minus EEGNet 50 Hz (paper run) (positive = beats EEGNet)

| Comparison | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg | Avg 95% CI | Subjects +/- |
|---|---:|---:|---:|---:|---:|---:|---:|
| EEGNet 50 Hz + fixed schedule | +1.66 (p <.001) | +1.85 (p <.001) | +2.39 (p <.001) | +2.35 (p <.001) | +2.06 (p <.001) | [+1.58, +2.55] | 18-0 |
| LEAN v2 (final) | +2.55 (p <.001) | +3.21 (p <.001) | +2.89 (p <.001) | +1.18 (p .005) | +2.46 (p <.001) | [+1.91, +3.01] | 18-0 |

## By joint group (window average)

| Group | Pairs | EEGNet 50 Hz + fixed schedule | LEAN v2 (final) | LEAN (paper run) | EEGNet 50 Hz (paper run) |
|---|---:|---:|---:|---:|---:|
| Distant | 11 | 67.26 | 67.55 | 58.59 | 64.77 |
| Adjacent | 12 | 64.29 | 64.69 | 56.29 | 62.53 |
| Same-joint | 5 | 64.40 | 65.00 | 56.86 | 62.57 |
| Overall | 28 | 65.48 | 65.87 | 57.30 | 63.41 |

## Per pair (window average)

| Pair | Group | EEGNet 50 Hz + fixed schedule | LEAN v2 (final) | LEAN (paper run) | EEGNet 50 Hz (paper run) |
|---|---:|---:|---:|---:|---:|
| HOC/WFE | Adjacent | 60.79 | 61.88 | 54.26 | 59.10 |
| HOC/WAA | Adjacent | 62.70 | 63.98 | 57.50 | 60.26 |
| HOC/EPS | Distant | 62.10 | 60.58 | 55.85 | 59.66 |
| HOC/EFE | Distant | 60.27 | 61.48 | 55.55 | 59.06 |
| HOC/SPS | Distant | 70.66 | 70.52 | 59.68 | 68.87 |
| HOC/SAA | Distant | 69.33 | 68.73 | 62.68 | 66.93 |
| HOC/SFE | Distant | 66.87 | 68.18 | 57.18 | 63.84 |
| WFE/WAA | Same-joint | 61.79 | 63.14 | 57.25 | 59.52 |
| WFE/EPS | Adjacent | 60.79 | 61.05 | 54.52 | 59.78 |
| WFE/EFE | Adjacent | 59.03 | 61.57 | 53.04 | 59.68 |
| WFE/SPS | Distant | 68.05 | 68.92 | 58.66 | 65.21 |
| WFE/SAA | Distant | 68.80 | 69.92 | 59.67 | 66.11 |
| WFE/SFE | Distant | 65.10 | 64.85 | 57.45 | 60.43 |
| WAA/EPS | Adjacent | 60.61 | 60.42 | 53.37 | 58.89 |
| WAA/EFE | Adjacent | 62.50 | 62.72 | 55.57 | 60.96 |
| WAA/SPS | Distant | 69.85 | 71.07 | 60.18 | 67.44 |
| WAA/SAA | Distant | 70.97 | 69.86 | 59.94 | 69.83 |
| WAA/SFE | Distant | 67.91 | 68.97 | 57.72 | 65.04 |
| EPS/EFE | Same-joint | 60.39 | 60.79 | 55.50 | 57.85 |
| EPS/SPS | Adjacent | 71.97 | 71.73 | 58.83 | 68.87 |
| EPS/SAA | Adjacent | 69.20 | 68.68 | 59.95 | 67.34 |
| EPS/SFE | Adjacent | 70.00 | 68.94 | 58.42 | 66.79 |
| EFE/SPS | Adjacent | 65.08 | 65.53 | 55.91 | 62.95 |
| EFE/SAA | Adjacent | 64.35 | 65.61 | 58.37 | 63.52 |
| EFE/SFE | Adjacent | 64.44 | 64.19 | 55.79 | 62.19 |
| SPS/SAA | Same-joint | 66.49 | 67.64 | 57.55 | 65.02 |
| SPS/SFE | Same-joint | 67.18 | 67.50 | 58.28 | 65.33 |
| SAA/SFE | Same-joint | 66.16 | 65.94 | 55.68 | 65.10 |

