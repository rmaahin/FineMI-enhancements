# EEGNet-8,2, 0.5-3 Hz (full mode, 28 pairs, 18 subjects)

Data: fixation-cross baseline (-4 to -2 s), motor imagery only (0 to 4 s after the trigger), 0.5-3 Hz. Windows start at imagery onset. EEGNet 250 Hz uses the full rate; EEGNet 50 Hz is decimated by 5.

## Parameters

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms |
|---|---:|---:|---:|---:|
| EEGNet 250 Hz (K1 125, pools 4/8, sep K 16) | 2,778 | 2,938 | 3,322 | 3,578 |
| EEGNet 50 Hz (K1 25, pools 2/4, sep K 8) | 1,818 | 1,946 | 2,234 | 2,458 |

## Mean accuracy over the pairs

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| EEGNet 250 Hz | 62.79 | 63.50 | 64.61 | 63.71 | 63.65 |
| EEGNet 50 Hz | 62.71 | 63.26 | 64.28 | 63.42 | 63.41 |
| LEAN | 60.06 | 58.29 | 55.67 | 55.17 | 57.30 |
| EEG Conformer | 62.19 | 61.59 | 60.99 | 60.86 | 61.41 |
| ShallowConvNet | 58.01 | 59.53 | 59.29 | 58.66 | 58.87 |

EEGNet 250 Hz SD over subjects (each subject's mean over the pairs): 800 ms 6.41, 1500 ms 6.25, 3000 ms 6.62, 4000 ms 6.03, Avg 6.11

EEGNet 50 Hz SD over subjects (each subject's mean over the pairs): 800 ms 6.42, 1500 ms 6.45, 3000 ms 6.37, 4000 ms 6.54, Avg 6.21

## By joint group (window average)

| Group | Pairs | EEGNet 250 Hz | EEGNet 50 Hz | LEAN | EEG Conformer | ShallowConvNet |
|---|---:|---:|---:|---:|---:|---:|
| Distant | 11 | 65.04 | 64.77 | 58.59 | 62.73 | 59.88 |
| Adjacent | 12 | 62.60 | 62.53 | 56.29 | 60.26 | 58.10 |
| Same-joint | 5 | 63.13 | 62.57 | 56.86 | 61.23 | 58.51 |
| Overall | 28 | 63.65 | 63.41 | 57.30 | 61.41 | 58.87 |

## EEGNet 250 Hz minus EEGNet 50 Hz, paired over subjects (pooled over 28 pairs)

| Window | Diff | 95% CI | Holm p | Subjects EEGNet 250 Hz better / worse |
|---|---:|---:|---:|---:|
| 800 ms | +0.09 | [-0.69, +0.86] | 1.000 | 10-8 |
| 1500 ms | +0.24 | [-0.57, +1.06] | 1.000 | 8-10 |
| 3000 ms | +0.33 | [-0.52, +1.18] | 1.000 | 11-7 |
| 4000 ms | +0.29 | [-0.39, +0.98] | 1.000 | 11-7 |
| Avg | +0.24 | [-0.17, +0.65] | 1.000 | 12-6 |

- Window average: EEGNet 250 Hz ahead in 16 of 28 pairs; largest lead WFE/SFE (+1.26), largest deficit WAA/EFE (-1.28)
- Pearson r between per-pair window averages: 0.98 (p <.001)

## EEGNet 250 Hz minus LEAN, paired over subjects (pooled over 28 pairs)

| Window | Diff | 95% CI | Holm p | Subjects EEGNet 250 Hz better / worse |
|---|---:|---:|---:|---:|
| 800 ms | +2.73 | [+1.71, +3.75] | <.001 | 16-2 |
| 1500 ms | +5.21 | [+3.59, +6.83] | <.001 | 17-1 |
| 3000 ms | +8.94 | [+5.96, +11.91] | <.001 | 18-0 |
| 4000 ms | +8.54 | [+5.80, +11.29] | <.001 | 18-0 |
| Avg | +6.35 | [+4.54, +8.17] | <.001 | 18-0 |

- Window average: EEGNet 250 Hz ahead in 28 of 28 pairs; largest lead EPS/SPS (+10.92), largest deficit HOC/WAA (+3.03)
- Pearson r between per-pair window averages: 0.81 (p <.001)

## EEGNet 250 Hz minus EEG Conformer, paired over subjects (pooled over 28 pairs)

| Window | Diff | 95% CI | Holm p | Subjects EEGNet 250 Hz better / worse |
|---|---:|---:|---:|---:|
| 800 ms | +0.60 | [-0.23, +1.44] | .145 | 11-7 |
| 1500 ms | +1.91 | [+0.82, +3.00] | .004 | 15-3 |
| 3000 ms | +3.62 | [+2.48, +4.76] | <.001 | 17-1 |
| 4000 ms | +2.85 | [+1.97, +3.72] | <.001 | 17-0 |
| Avg | +2.25 | [+1.48, +3.01] | <.001 | 17-1 |

- Window average: EEGNet 250 Hz ahead in 28 of 28 pairs; largest lead HOC/SFE (+4.19), largest deficit SPS/SFE (+0.34)
- Pearson r between per-pair window averages: 0.97 (p <.001)

## EEGNet 250 Hz minus ShallowConvNet, paired over subjects (pooled over 28 pairs)

| Window | Diff | 95% CI | Holm p | Subjects EEGNet 250 Hz better / worse |
|---|---:|---:|---:|---:|
| 800 ms | +4.78 | [+3.31, +6.26] | <.001 | 18-0 |
| 1500 ms | +3.97 | [+1.91, +6.04] | <.001 | 14-4 |
| 3000 ms | +5.32 | [+2.90, +7.74] | <.001 | 16-2 |
| 4000 ms | +5.05 | [+2.94, +7.15] | <.001 | 17-1 |
| Avg | +4.78 | [+2.93, +6.63] | <.001 | 16-2 |

- Window average: EEGNet 250 Hz ahead in 28 of 28 pairs; largest lead WAA/SAA (+9.01), largest deficit HOC/WAA (+1.48)
- Pearson r between per-pair window averages: 0.89 (p <.001)

## EEGNet 50 Hz minus LEAN, paired over subjects (pooled over 28 pairs)

| Window | Diff | 95% CI | Holm p | Subjects EEGNet 50 Hz better / worse |
|---|---:|---:|---:|---:|
| 800 ms | +2.65 | [+1.36, +3.93] | <.001 | 14-4 |
| 1500 ms | +4.96 | [+3.29, +6.64] | <.001 | 17-1 |
| 3000 ms | +8.60 | [+5.60, +11.61] | <.001 | 17-1 |
| 4000 ms | +8.25 | [+5.46, +11.04] | <.001 | 17-1 |
| Avg | +6.12 | [+4.21, +8.02] | <.001 | 18-0 |

- Window average: EEGNet 50 Hz ahead in 28 of 28 pairs; largest lead EPS/SPS (+10.04), largest deficit WFE/WAA (+2.27)
- Pearson r between per-pair window averages: 0.81 (p <.001)

## EEGNet 50 Hz minus EEG Conformer, paired over subjects (pooled over 28 pairs)

| Window | Diff | 95% CI | Holm p | Subjects EEGNet 50 Hz better / worse |
|---|---:|---:|---:|---:|
| 800 ms | +0.52 | [-0.33, +1.37] | .214 | 12-6 |
| 1500 ms | +1.67 | [+0.59, +2.74] | .009 | 13-5 |
| 3000 ms | +3.29 | [+2.12, +4.45] | <.001 | 16-2 |
| 4000 ms | +2.56 | [+1.61, +3.50] | <.001 | 18-0 |
| Avg | +2.01 | [+1.17, +2.84] | <.001 | 16-2 |

- Window average: EEGNet 50 Hz ahead in 28 of 28 pairs; largest lead HOC/SPS (+3.60), largest deficit EPS/EFE (+0.14)
- Pearson r between per-pair window averages: 0.96 (p <.001)

## EEGNet 50 Hz minus ShallowConvNet, paired over subjects (pooled over 28 pairs)

| Window | Diff | 95% CI | Holm p | Subjects EEGNet 50 Hz better / worse |
|---|---:|---:|---:|---:|
| 800 ms | +4.70 | [+2.94, +6.46] | <.001 | 17-1 |
| 1500 ms | +3.73 | [+1.85, +5.61] | <.001 | 16-2 |
| 3000 ms | +4.99 | [+2.56, +7.41] | <.001 | 15-3 |
| 4000 ms | +4.75 | [+2.75, +6.76] | <.001 | 17-1 |
| Avg | +4.54 | [+2.71, +6.37] | <.001 | 16-2 |

- Window average: EEGNet 50 Hz ahead in 28 of 28 pairs; largest lead WAA/SAA (+9.04), largest deficit HOC/WAA (+1.21)
- Pearson r between per-pair window averages: 0.90 (p <.001)

## Per pair (sorted by the first EEGNet model's window average)

| Pair | Group | EEGNet 250 Hz 800 ms | EEGNet 250 Hz 1500 ms | EEGNet 250 Hz 3000 ms | EEGNet 250 Hz 4000 ms | EEGNet 250 Hz Avg | EEGNet 50 Hz 800 ms | EEGNet 50 Hz 1500 ms | EEGNet 50 Hz 3000 ms | EEGNet 50 Hz 4000 ms | EEGNet 50 Hz Avg | LEAN Avg | EEG Conformer Avg | ShallowConvNet Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WAA/SAA | Distant | 68.67 | 68.66 | 70.15 | 71.74 | 69.80 | 69.67 | 71.47 | 69.50 | 68.70 | 69.83 | 59.94 | 66.27 | 60.79 |
| EPS/SPS | Adjacent | 70.65 | 69.17 | 69.54 | 69.65 | 69.75 | 65.35 | 69.90 | 69.31 | 70.93 | 68.87 | 58.83 | 67.34 | 61.98 |
| EPS/SAA | Adjacent | 66.33 | 69.21 | 69.20 | 69.22 | 68.49 | 65.10 | 68.60 | 68.27 | 67.38 | 67.34 | 59.95 | 64.94 | 59.54 |
| HOC/SPS | Distant | 68.72 | 69.10 | 68.56 | 67.23 | 68.40 | 68.67 | 68.83 | 70.12 | 67.84 | 68.87 | 59.68 | 65.27 | 61.41 |
| EPS/SFE | Adjacent | 66.16 | 65.40 | 69.15 | 68.26 | 67.24 | 64.88 | 67.95 | 66.69 | 67.65 | 66.79 | 58.42 | 64.05 | 60.01 |
| HOC/SAA | Distant | 65.28 | 68.01 | 68.62 | 66.52 | 67.11 | 66.64 | 65.76 | 69.07 | 66.23 | 66.93 | 62.68 | 66.49 | 62.70 |
| WAA/SPS | Distant | 66.87 | 66.94 | 68.15 | 65.74 | 66.92 | 67.70 | 66.71 | 68.34 | 67.02 | 67.44 | 60.18 | 65.93 | 61.20 |
| WFE/SPS | Distant | 66.40 | 66.60 | 67.79 | 64.46 | 66.31 | 66.60 | 64.99 | 65.52 | 63.74 | 65.21 | 58.66 | 64.34 | 61.19 |
| SAA/SFE | Same-joint | 64.88 | 66.16 | 67.49 | 66.46 | 66.25 | 64.66 | 64.86 | 65.96 | 64.92 | 65.10 | 55.68 | 64.39 | 59.39 |
| WFE/SAA | Distant | 64.71 | 65.89 | 66.91 | 65.90 | 65.85 | 65.59 | 65.77 | 66.90 | 66.16 | 66.11 | 59.67 | 64.23 | 61.16 |
| WAA/SFE | Distant | 64.51 | 66.06 | 66.75 | 65.68 | 65.75 | 63.71 | 65.42 | 64.91 | 66.12 | 65.04 | 57.72 | 62.09 | 60.38 |
| SPS/SAA | Same-joint | 63.56 | 66.15 | 65.72 | 65.02 | 65.11 | 64.34 | 65.32 | 65.68 | 64.74 | 65.02 | 57.55 | 62.29 | 61.36 |
| SPS/SFE | Same-joint | 63.20 | 65.48 | 65.62 | 65.95 | 65.06 | 64.26 | 64.52 | 67.20 | 65.35 | 65.33 | 58.28 | 64.73 | 59.94 |
| HOC/SFE | Distant | 63.48 | 65.02 | 66.51 | 64.94 | 64.98 | 63.58 | 62.77 | 65.43 | 63.58 | 63.84 | 57.18 | 60.79 | 59.77 |
| EFE/SAA | Adjacent | 63.77 | 63.63 | 64.64 | 63.42 | 63.87 | 63.18 | 61.94 | 66.12 | 62.85 | 63.52 | 58.37 | 61.22 | 59.46 |
| EFE/SPS | Adjacent | 61.49 | 61.81 | 63.86 | 63.50 | 62.67 | 61.19 | 65.00 | 64.65 | 60.98 | 62.95 | 55.91 | 60.78 | 60.37 |
| EFE/SFE | Adjacent | 59.18 | 61.29 | 64.84 | 63.35 | 62.16 | 60.88 | 62.06 | 62.94 | 62.89 | 62.19 | 55.79 | 60.11 | 57.97 |
| WFE/SFE | Distant | 61.42 | 60.13 | 62.24 | 62.95 | 61.69 | 59.39 | 60.24 | 60.02 | 62.07 | 60.43 | 57.45 | 59.60 | 57.61 |
| HOC/WAA | Adjacent | 60.36 | 59.38 | 61.67 | 60.73 | 60.53 | 61.67 | 59.06 | 60.18 | 60.15 | 60.26 | 57.50 | 58.78 | 59.05 |
| WFE/WAA | Same-joint | 57.87 | 62.28 | 61.37 | 60.02 | 60.38 | 55.89 | 61.09 | 60.59 | 60.53 | 59.52 | 57.25 | 57.06 | 57.18 |
| WFE/EPS | Adjacent | 58.46 | 58.96 | 61.98 | 60.39 | 59.95 | 59.03 | 59.33 | 60.70 | 60.05 | 59.78 | 54.52 | 57.16 | 56.34 |
| WAA/EFE | Adjacent | 59.79 | 61.10 | 59.23 | 58.58 | 59.67 | 59.95 | 60.86 | 61.50 | 61.52 | 60.96 | 55.57 | 58.03 | 57.12 |
| HOC/EFE | Distant | 59.24 | 60.23 | 59.69 | 58.84 | 59.50 | 58.46 | 58.15 | 60.39 | 59.24 | 59.06 | 55.55 | 57.94 | 56.34 |
| WFE/EFE | Adjacent | 59.23 | 58.76 | 59.94 | 58.86 | 59.20 | 59.78 | 57.88 | 60.51 | 60.57 | 59.68 | 53.04 | 56.85 | 55.33 |
| HOC/EPS | Distant | 58.41 | 57.92 | 61.33 | 59.01 | 59.17 | 59.57 | 58.59 | 61.93 | 58.54 | 59.66 | 55.85 | 57.12 | 56.15 |
| HOC/WFE | Adjacent | 58.25 | 56.80 | 61.13 | 59.98 | 59.04 | 58.59 | 56.84 | 59.71 | 61.26 | 59.10 | 54.26 | 56.86 | 54.88 |
| EPS/EFE | Same-joint | 59.81 | 59.55 | 57.54 | 58.40 | 58.83 | 59.44 | 59.05 | 56.92 | 56.00 | 57.85 | 55.50 | 57.71 | 54.67 |
| WAA/EPS | Adjacent | 57.41 | 58.40 | 59.46 | 59.05 | 58.58 | 57.99 | 58.28 | 60.70 | 58.60 | 58.89 | 53.37 | 57.05 | 55.11 |

