# ShallowConvNet, 0.5-3 Hz (full mode, 28 pairs, 18 subjects)

Data: fixation-cross baseline (-4 to -2 s), motor imagery only (0 to 4 s after the trigger), 0.5-3 Hz, 250 Hz. Windows start at imagery onset.

## Parameters

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms |
|---|---:|---:|---:|---:|
| ShallowConvNet | 100,882 | 101,842 | 103,842 | 105,202 |

## Mean accuracy over the pairs

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| ShallowConvNet | 58.01 | 59.53 | 59.29 | 58.66 | 58.87 |
| LEAN | 60.06 | 58.29 | 55.67 | 55.17 | 57.30 |
| EEG Conformer | 62.19 | 61.59 | 60.99 | 60.86 | 61.41 |

ShallowConvNet SD over subjects (each subject's mean over the pairs): 800 ms 5.68, 1500 ms 6.24, 3000 ms 6.51, 4000 ms 6.39, Avg 5.81

## By joint group (window average)

| Group | Pairs | ShallowConvNet | LEAN | EEG Conformer |
|---|---:|---:|---:|---:|
| Distant | 11 | 59.88 | 58.59 | 62.73 |
| Adjacent | 12 | 58.10 | 56.29 | 60.26 |
| Same-joint | 5 | 58.51 | 56.86 | 61.23 |
| Overall | 28 | 58.87 | 57.30 | 61.41 |

## ShallowConvNet minus LEAN, paired over subjects (pooled over 28 pairs)

| Window | Diff | 95% CI | Holm p | Subjects ShallowConvNet better / worse |
|---|---:|---:|---:|---:|
| 800 ms | -2.05 | [-3.42, -0.68] | .017 | 6-12 |
| 1500 ms | +1.24 | [-0.16, +2.63] | .080 | 10-8 |
| 3000 ms | +3.61 | [+1.31, +5.92] | .016 | 16-2 |
| 4000 ms | +3.50 | [+1.56, +5.43] | .007 | 16-2 |
| Avg | +1.57 | [+0.18, +2.97] | .058 | 14-4 |

- Window average: ShallowConvNet ahead in 25 of 28 pairs; largest lead EFE/SPS (+4.46), largest deficit EPS/EFE (-0.83)
- Pearson r between per-pair window averages: 0.85 (p <.001)

## ShallowConvNet minus EEG Conformer, paired over subjects (pooled over 28 pairs)

| Window | Diff | 95% CI | Holm p | Subjects ShallowConvNet better / worse |
|---|---:|---:|---:|---:|
| 800 ms | -4.18 | [-5.59, -2.77] | <.001 | 1-17 |
| 1500 ms | -2.06 | [-3.70, -0.43] | .033 | 4-14 |
| 3000 ms | -1.70 | [-3.54, +0.14] | .068 | 7-11 |
| 4000 ms | -2.20 | [-3.77, -0.63] | .027 | 4-14 |
| Avg | -2.54 | [-3.95, -1.12] | .006 | 4-14 |

- Window average: ShallowConvNet ahead in 2 of 28 pairs; largest lead HOC/WAA (+0.27), largest deficit WAA/SAA (-5.48)
- Pearson r between per-pair window averages: 0.90 (p <.001)

## Per pair

| Pair | Group | Shallow 800 ms | Shallow 1500 ms | Shallow 3000 ms | Shallow 4000 ms | Shallow Avg | LEAN Avg | EEG Conformer Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HOC/SAA | Distant | 60.46 | 64.38 | 63.19 | 62.77 | 62.70 | 62.68 | 66.49 |
| EPS/SPS | Adjacent | 62.65 | 63.40 | 61.99 | 59.88 | 61.98 | 58.83 | 67.34 |
| HOC/SPS | Distant | 61.84 | 61.40 | 60.52 | 61.88 | 61.41 | 59.68 | 65.27 |
| SPS/SAA | Same-joint | 60.79 | 62.10 | 63.21 | 59.35 | 61.36 | 57.55 | 62.29 |
| WAA/SPS | Distant | 59.67 | 62.07 | 60.86 | 62.21 | 61.20 | 60.18 | 65.93 |
| WFE/SPS | Distant | 60.95 | 60.56 | 60.83 | 62.43 | 61.19 | 58.66 | 64.34 |
| WFE/SAA | Distant | 59.57 | 62.38 | 63.41 | 59.27 | 61.16 | 59.67 | 64.23 |
| WAA/SAA | Distant | 60.69 | 62.99 | 61.05 | 58.43 | 60.79 | 59.94 | 66.27 |
| WAA/SFE | Distant | 58.86 | 61.63 | 60.87 | 60.16 | 60.38 | 57.72 | 62.09 |
| EFE/SPS | Adjacent | 60.70 | 60.12 | 59.65 | 61.02 | 60.37 | 55.91 | 60.78 |
| EPS/SFE | Adjacent | 58.80 | 61.02 | 60.32 | 59.90 | 60.01 | 58.42 | 64.05 |
| SPS/SFE | Same-joint | 59.60 | 60.16 | 60.68 | 59.31 | 59.94 | 58.28 | 64.73 |
| HOC/SFE | Distant | 58.68 | 60.75 | 59.81 | 59.83 | 59.77 | 57.18 | 60.79 |
| EPS/SAA | Adjacent | 60.98 | 60.59 | 60.26 | 56.33 | 59.54 | 59.95 | 64.94 |
| EFE/SAA | Adjacent | 57.55 | 60.19 | 60.48 | 59.62 | 59.46 | 58.37 | 61.22 |
| SAA/SFE | Same-joint | 57.32 | 60.21 | 59.01 | 61.00 | 59.39 | 55.68 | 64.39 |
| HOC/WAA | Adjacent | 55.99 | 59.44 | 61.11 | 59.66 | 59.05 | 57.50 | 58.78 |
| EFE/SFE | Adjacent | 57.22 | 58.70 | 58.83 | 57.13 | 57.97 | 55.79 | 60.11 |
| WFE/SFE | Distant | 54.63 | 58.19 | 60.00 | 57.61 | 57.61 | 57.45 | 59.60 |
| WFE/WAA | Same-joint | 53.04 | 57.15 | 58.76 | 59.76 | 57.18 | 57.25 | 57.06 |
| WAA/EFE | Adjacent | 56.37 | 56.17 | 58.36 | 57.57 | 57.12 | 55.57 | 58.03 |
| WFE/EPS | Adjacent | 56.28 | 57.32 | 56.10 | 55.67 | 56.34 | 54.52 | 57.16 |
| HOC/EFE | Distant | 55.42 | 57.69 | 55.28 | 56.98 | 56.34 | 55.55 | 57.94 |
| HOC/EPS | Distant | 55.83 | 56.79 | 56.58 | 55.41 | 56.15 | 55.85 | 57.12 |
| WFE/EFE | Adjacent | 54.21 | 55.65 | 55.09 | 56.36 | 55.33 | 53.04 | 56.85 |
| WAA/EPS | Adjacent | 54.85 | 54.71 | 55.44 | 55.43 | 55.11 | 53.37 | 57.05 |
| HOC/WFE | Adjacent | 54.80 | 55.61 | 54.50 | 54.62 | 54.88 | 54.26 | 56.86 |
| EPS/EFE | Same-joint | 56.45 | 55.45 | 53.91 | 52.88 | 54.67 | 55.50 | 57.71 |

