# ShallowConvNet, 0.5-3 Hz (smoke mode, 3 pairs, 2 subjects)

Data: fixation-cross baseline (-4 to -2 s), motor imagery only (0 to 4 s after the trigger), 0.5-3 Hz, 250 Hz. Windows start at imagery onset.

## Parameters

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms |
|---|---:|---:|---:|---:|
| ShallowConvNet | 100,882 | 101,842 | 103,842 | 105,202 |

## Mean accuracy over the pairs

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| ShallowConvNet | 50.95 | 49.17 | 51.85 | 51.53 | 50.87 |
| LEAN | 52.06 | 50.49 | 46.90 | 49.61 | 49.76 |
| EEG Conformer | 60.09 | 58.68 | 54.38 | 53.91 | 56.77 |

ShallowConvNet SD over subjects (each subject's mean over the pairs): 800 ms 3.96, 1500 ms 1.18, 3000 ms 2.10, 4000 ms 3.73, Avg 2.15

## By joint group (window average)

| Group | Pairs | ShallowConvNet | LEAN | EEG Conformer |
|---|---:|---:|---:|---:|
| Distant | 1 | 50.31 | 46.82 | 56.27 |
| Adjacent | 2 | 51.15 | 51.23 | 57.01 |
| Overall | 3 | 50.87 | 49.76 | 56.77 |

## ShallowConvNet minus LEAN, paired over subjects (pooled over 3 pairs)

| Window | Diff | 95% CI | Holm p | Subjects ShallowConvNet better / worse |
|---|---:|---:|---:|---:|
| 800 ms | -1.11 | [-15.23, +13.01] | .886 | 0-1 |
| 1500 ms | -1.32 | [-12.79, +10.15] | .886 | 0-2 |
| 3000 ms | +4.95 | [-6.22, +16.13] | .559 | 2-0 |
| 4000 ms | +1.92 | [-6.61, +10.45] | .856 | 2-0 |
| Avg | +1.11 | [-5.95, +8.17] | .886 | 2-0 |

- Window average: ShallowConvNet ahead in 2 of 3 pairs; largest lead WAA/SAA (+3.49), largest deficit EPS/SPS (-1.25)
- Pearson r between per-pair window averages: 0.99 (p .083)

## ShallowConvNet minus EEG Conformer, paired over subjects (pooled over 3 pairs)

| Window | Diff | 95% CI | Holm p | Subjects ShallowConvNet better / worse |
|---|---:|---:|---:|---:|
| 800 ms | -9.14 | [-35.91, +17.62] | .549 | 0-2 |
| 1500 ms | -9.51 | [-31.57, +12.55] | .549 | 0-2 |
| 3000 ms | -2.52 | [-8.11, +3.07] | .549 | 0-2 |
| 4000 ms | -2.38 | [-51.50, +46.73] | .648 | 1-1 |
| Avg | -5.89 | [-28.98, +17.20] | .549 | 0-2 |

- Window average: ShallowConvNet ahead in 0 of 3 pairs; largest lead HOC/WFE (-2.99), largest deficit EPS/SPS (-8.73)
- Pearson r between per-pair window averages: 0.48 (p .682)

## Per pair

| Pair | Group | Shallow 800 ms | Shallow 1500 ms | Shallow 3000 ms | Shallow 4000 ms | Shallow Avg | LEAN Avg | EEG Conformer Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EPS/SPS | Adjacent | 52.57 | 47.57 | 52.15 | 53.06 | 51.34 | 52.59 | 60.07 |
| HOC/WFE | Adjacent | 50.69 | 51.74 | 49.58 | 51.88 | 50.97 | 49.88 | 53.96 |
| WAA/SAA | Distant | 49.58 | 48.19 | 53.82 | 49.65 | 50.31 | 46.82 | 56.27 |

Note: smoke-mode numbers only prove the pipeline runs; they say nothing about accuracy.

