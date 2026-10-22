# ShallowConvNet, 8-30 Hz (full mode, 28 pairs, 18 subjects)

Data: fixation-cross baseline (-4 to -2 s), motor imagery only (0 to 4 s after the trigger), 8-30 Hz, 250 Hz. Windows start at imagery onset.

## Parameters

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms |
|---|---:|---:|---:|---:|
| ShallowConvNet | 100,882 | 101,842 | 103,842 | 105,202 |

## Mean accuracy over the pairs

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| ShallowConvNet | 56.70 | 57.30 | 57.26 | 57.07 | 57.08 |
| ShallowConvNet 0.5-3 Hz | 58.01 | 59.53 | 59.29 | 58.66 | 58.87 |

ShallowConvNet SD over subjects (each subject's mean over the pairs): 800 ms 4.93, 1500 ms 4.45, 3000 ms 5.32, 4000 ms 5.05, Avg 4.65

## By joint group (window average)

| Group | Pairs | ShallowConvNet | ShallowConvNet 0.5-3 Hz |
|---|---:|---:|---:|
| Distant | 11 | 59.08 | 59.88 |
| Adjacent | 12 | 55.84 | 58.10 |
| Same-joint | 5 | 55.68 | 58.51 |
| Overall | 28 | 57.08 | 58.87 |

## ShallowConvNet minus ShallowConvNet 0.5-3 Hz, paired over subjects (pooled over 28 pairs)

| Window | Diff | 95% CI | Holm p | Subjects ShallowConvNet better / worse |
|---|---:|---:|---:|---:|
| 800 ms | -1.30 | [-3.19, +0.58] | .707 | 7-11 |
| 1500 ms | -2.23 | [-5.28, +0.82] | .707 | 7-11 |
| 3000 ms | -2.03 | [-5.59, +1.53] | .707 | 9-9 |
| 4000 ms | -1.59 | [-5.10, +1.93] | .707 | 9-9 |
| Avg | -1.79 | [-4.62, +1.04] | .707 | 10-8 |

- Window average: ShallowConvNet ahead in 4 of 28 pairs; largest lead HOC/SFE (+1.87), largest deficit EFE/SAA (-5.93)
- Pearson r between per-pair window averages: 0.69 (p <.001)

## Per pair

| Pair | Group | Shallow 800 ms | Shallow 1500 ms | Shallow 3000 ms | Shallow 4000 ms | Shallow Avg | ShallowConvNet 0.5-3 Hz Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| HOC/SPS | Distant | 61.17 | 62.58 | 63.13 | 62.06 | 62.24 | 61.41 |
| HOC/SAA | Distant | 61.17 | 62.12 | 61.84 | 61.50 | 61.66 | 62.70 |
| HOC/SFE | Distant | 60.34 | 62.57 | 62.13 | 61.51 | 61.64 | 59.77 |
| WFE/SPS | Distant | 60.35 | 61.04 | 61.05 | 60.15 | 60.65 | 61.19 |
| WAA/SAA | Distant | 59.71 | 60.39 | 59.04 | 60.90 | 60.01 | 60.79 |
| EPS/SPS | Adjacent | 58.72 | 59.75 | 58.66 | 58.41 | 58.89 | 61.98 |
| WAA/SFE | Distant | 56.85 | 58.52 | 58.87 | 58.54 | 58.19 | 60.38 |
| WAA/SPS | Distant | 58.96 | 56.47 | 57.62 | 58.20 | 57.81 | 61.20 |
| WFE/SAA | Distant | 57.78 | 58.08 | 57.01 | 58.19 | 57.76 | 61.16 |
| SPS/SFE | Same-joint | 57.46 | 58.53 | 56.68 | 57.01 | 57.42 | 59.94 |
| HOC/EFE | Distant | 57.08 | 58.27 | 56.30 | 57.64 | 57.32 | 56.34 |
| HOC/WAA | Adjacent | 56.29 | 58.52 | 57.38 | 56.98 | 57.29 | 59.05 |
| EPS/SAA | Adjacent | 58.11 | 57.05 | 57.80 | 56.05 | 57.25 | 59.54 |
| WFE/SFE | Distant | 56.26 | 58.50 | 56.85 | 57.39 | 57.25 | 57.61 |
| EPS/SFE | Adjacent | 58.73 | 57.40 | 57.10 | 55.15 | 57.10 | 60.01 |
| WFE/EFE | Adjacent | 55.59 | 55.82 | 56.50 | 58.27 | 56.54 | 55.33 |
| SPS/SAA | Same-joint | 56.91 | 55.55 | 56.66 | 55.90 | 56.25 | 61.36 |
| EFE/SPS | Adjacent | 57.67 | 55.82 | 54.68 | 56.16 | 56.08 | 60.37 |
| WFE/WAA | Same-joint | 53.43 | 54.64 | 57.35 | 56.91 | 55.58 | 57.18 |
| WFE/EPS | Adjacent | 54.31 | 56.13 | 56.61 | 54.88 | 55.48 | 56.34 |
| EFE/SFE | Adjacent | 54.92 | 56.61 | 54.01 | 56.28 | 55.46 | 57.97 |
| HOC/EPS | Distant | 52.55 | 55.86 | 57.48 | 55.60 | 55.37 | 56.15 |
| WAA/EFE | Adjacent | 53.77 | 54.38 | 57.08 | 55.61 | 55.21 | 57.12 |
| SAA/SFE | Same-joint | 55.76 | 55.49 | 54.43 | 54.00 | 54.92 | 59.39 |
| HOC/WFE | Adjacent | 53.12 | 53.31 | 56.22 | 54.71 | 54.34 | 54.88 |
| EPS/EFE | Same-joint | 54.32 | 54.39 | 54.76 | 53.46 | 54.23 | 54.67 |
| EFE/SAA | Adjacent | 54.97 | 52.96 | 53.15 | 53.05 | 53.53 | 59.46 |
| WAA/EPS | Adjacent | 51.39 | 53.65 | 52.96 | 53.50 | 52.87 | 55.11 |

