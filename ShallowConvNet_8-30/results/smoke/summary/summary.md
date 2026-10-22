# ShallowConvNet, 8-30 Hz (smoke mode, 3 pairs, 2 subjects)

Data: fixation-cross baseline (-4 to -2 s), motor imagery only (0 to 4 s after the trigger), 8-30 Hz, 250 Hz. Windows start at imagery onset.

## Parameters

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms |
|---|---:|---:|---:|---:|
| ShallowConvNet | 100,882 | 101,842 | 103,842 | 105,202 |

## Mean accuracy over the pairs

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| ShallowConvNet | 49.47 | 50.21 | 50.93 | 51.67 | 50.57 |
| ShallowConvNet 0.5-3 Hz | 50.95 | 49.17 | 51.85 | 51.53 | 50.87 |

ShallowConvNet SD over subjects (each subject's mean over the pairs): 800 ms 3.37, 1500 ms 0.29, 3000 ms 1.31, 4000 ms 2.36, Avg 1.69

## By joint group (window average)

| Group | Pairs | ShallowConvNet | ShallowConvNet 0.5-3 Hz |
|---|---:|---:|---:|
| Distant | 1 | 52.08 | 50.31 |
| Adjacent | 2 | 49.81 | 51.15 |
| Overall | 3 | 50.57 | 50.87 |

## ShallowConvNet minus ShallowConvNet 0.5-3 Hz, paired over subjects (pooled over 3 pairs)

| Window | Diff | 95% CI | Holm p | Subjects ShallowConvNet better / worse |
|---|---:|---:|---:|---:|
| 800 ms | -1.48 | [-67.37, +64.40] | 1.000 | 1-1 |
| 1500 ms | +1.04 | [-12.19, +14.28] | 1.000 | 1-0 |
| 3000 ms | -0.93 | [-31.51, +29.66] | 1.000 | 1-1 |
| 4000 ms | +0.14 | [-54.57, +54.85] | 1.000 | 1-1 |
| Avg | -0.31 | [-34.79, +34.18] | 1.000 | 1-1 |

- Window average: ShallowConvNet ahead in 1 of 3 pairs; largest lead WAA/SAA (+1.77), largest deficit EPS/SPS (-1.41)
- Pearson r between per-pair window averages: -0.90 (p .287)

## Per pair

| Pair | Group | Shallow 800 ms | Shallow 1500 ms | Shallow 3000 ms | Shallow 4000 ms | Shallow Avg | ShallowConvNet 0.5-3 Hz Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| WAA/SAA | Distant | 51.11 | 50.00 | 52.78 | 54.44 | 52.08 | 50.31 |
| EPS/SPS | Adjacent | 49.17 | 50.00 | 50.00 | 50.56 | 49.93 | 51.34 |
| HOC/WFE | Adjacent | 48.12 | 50.62 | 50.00 | 50.00 | 49.69 | 50.97 |

Note: smoke-mode numbers only prove the pipeline runs; they say nothing about accuracy.

