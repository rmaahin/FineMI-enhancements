# EEGNet-8,2, 0.5-3 Hz (smoke mode, 3 pairs, 2 subjects)

Data: fixation-cross baseline (-4 to -2 s), motor imagery only (0 to 4 s after the trigger), 0.5-3 Hz. Windows start at imagery onset. EEGNet 250 Hz uses the full rate; EEGNet 50 Hz is decimated by 5.

## Parameters

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms |
|---|---:|---:|---:|---:|
| EEGNet 250 Hz (K1 125, pools 4/8, sep K 16) | 2,778 | 2,938 | 3,322 | 3,578 |
| EEGNet 50 Hz (K1 25, pools 2/4, sep K 8) | 1,818 | 1,946 | 2,234 | 2,458 |

## Mean accuracy over the pairs

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| EEGNet 250 Hz | 56.94 | 56.81 | 55.12 | 57.87 | 56.68 |
| EEGNet 50 Hz | 54.88 | 50.95 | 57.50 | 59.54 | 55.72 |
| LEAN | 52.06 | 50.49 | 46.90 | 49.61 | 49.76 |
| EEG Conformer | 60.09 | 58.68 | 54.38 | 53.91 | 56.77 |
| ShallowConvNet | 50.95 | 49.17 | 51.85 | 51.53 | 50.87 |

EEGNet 250 Hz SD over subjects (each subject's mean over the pairs): 800 ms 11.39, 1500 ms 1.37, 3000 ms 0.43, 4000 ms 5.37, Avg 3.95

EEGNet 50 Hz SD over subjects (each subject's mean over the pairs): 800 ms 4.29, 1500 ms 3.96, 3000 ms 5.89, 4000 ms 3.01, Avg 4.29

## By joint group (window average)

| Group | Pairs | EEGNet 250 Hz | EEGNet 50 Hz | LEAN | EEG Conformer | ShallowConvNet |
|---|---:|---:|---:|---:|---:|---:|
| Distant | 1 | 58.78 | 54.38 | 46.82 | 56.27 | 50.31 |
| Adjacent | 2 | 55.63 | 56.39 | 51.23 | 57.01 | 51.15 |
| Overall | 3 | 56.68 | 55.72 | 49.76 | 56.77 | 50.87 |

## EEGNet 250 Hz minus EEGNet 50 Hz, paired over subjects (pooled over 3 pairs)

| Window | Diff | 95% CI | Holm p | Subjects EEGNet 250 Hz better / worse |
|---|---:|---:|---:|---:|
| 800 ms | +2.06 | [-61.76, +65.89] | 1.000 | 1-1 |
| 1500 ms | +5.86 | [-42.09, +53.80] | 1.000 | 2-0 |
| 3000 ms | -2.38 | [-51.50, +46.73] | 1.000 | 1-1 |
| 4000 ms | -1.67 | [-22.84, +19.51] | 1.000 | 0-1 |
| Avg | +0.97 | [-2.05, +3.98] | .766 | 2-0 |

- Window average: EEGNet 250 Hz ahead in 2 of 3 pairs; largest lead WAA/SAA (+4.41), largest deficit HOC/WFE (-1.60)
- Pearson r between per-pair window averages: -0.52 (p .654)

## EEGNet 250 Hz minus LEAN, paired over subjects (pooled over 3 pairs)

| Window | Diff | 95% CI | Holm p | Subjects EEGNet 250 Hz better / worse |
|---|---:|---:|---:|---:|
| 800 ms | +4.88 | [-76.00, +85.77] | .583 | 1-1 |
| 1500 ms | +6.32 | [-3.39, +16.03] | .230 | 2-0 |
| 3000 ms | +8.22 | [+4.39, +12.04] | .117 | 2-0 |
| 4000 ms | +8.26 | [+2.09, +14.44] | .150 | 2-0 |
| Avg | +6.92 | [-16.31, +30.16] | .329 | 2-0 |

- Window average: EEGNet 250 Hz ahead in 3 of 3 pairs; largest lead WAA/SAA (+11.96), largest deficit EPS/SPS (+4.38)
- Pearson r between per-pair window averages: -0.44 (p .713)

## EEGNet 250 Hz minus EEG Conformer, paired over subjects (pooled over 3 pairs)

| Window | Diff | 95% CI | Holm p | Subjects EEGNet 250 Hz better / worse |
|---|---:|---:|---:|---:|
| 800 ms | -3.15 | [-43.15, +36.85] | 1.000 | 0-1 |
| 1500 ms | -1.88 | [-25.70, +21.95] | 1.000 | 0-1 |
| 3000 ms | +0.74 | [-8.67, +10.15] | 1.000 | 1-0 |
| 4000 ms | +3.96 | [-30.45, +38.37] | 1.000 | 2-0 |
| Avg | -0.08 | [-6.99, +6.83] | 1.000 | 1-1 |

- Window average: EEGNet 250 Hz ahead in 2 of 3 pairs; largest lead WAA/SAA (+2.52), largest deficit EPS/SPS (-3.11)
- Pearson r between per-pair window averages: 0.47 (p .688)

## EEGNet 250 Hz minus ShallowConvNet, paired over subjects (pooled over 3 pairs)

| Window | Diff | 95% CI | Holm p | Subjects EEGNet 250 Hz better / worse |
|---|---:|---:|---:|---:|
| 800 ms | +6.00 | [-60.77, +72.76] | .460 | 2-0 |
| 1500 ms | +7.64 | [+5.87, +9.40] | .058 | 2-0 |
| 3000 ms | +3.26 | [-11.74, +18.26] | .460 | 2-0 |
| 4000 ms | +6.34 | [-8.36, +21.05] | .460 | 2-0 |
| Avg | +5.81 | [-10.37, +21.99] | .460 | 2-0 |

- Window average: EEGNet 250 Hz ahead in 3 of 3 pairs; largest lead WAA/SAA (+8.47), largest deficit HOC/WFE (+3.33)
- Pearson r between per-pair window averages: -0.55 (p .630)

## EEGNet 50 Hz minus LEAN, paired over subjects (pooled over 3 pairs)

| Window | Diff | 95% CI | Holm p | Subjects EEGNet 50 Hz better / worse |
|---|---:|---:|---:|---:|
| 800 ms | +2.82 | [-14.24, +19.88] | .826 | 2-0 |
| 1500 ms | +0.46 | [-57.19, +58.11] | .935 | 1-1 |
| 3000 ms | +10.60 | [-34.69, +55.90] | .826 | 2-0 |
| 4000 ms | +9.93 | [-5.07, +24.93] | .377 | 2-0 |
| Avg | +5.95 | [-20.30, +32.21] | .826 | 2-0 |

- Window average: EEGNet 50 Hz ahead in 3 of 3 pairs; largest lead WAA/SAA (+7.55), largest deficit EPS/SPS (+4.29)
- Pearson r between per-pair window averages: 1.00 (p .059)

## EEGNet 50 Hz minus EEG Conformer, paired over subjects (pooled over 3 pairs)

| Window | Diff | 95% CI | Holm p | Subjects EEGNet 50 Hz better / worse |
|---|---:|---:|---:|---:|
| 800 ms | -5.21 | [-29.03, +18.62] | .766 | 0-2 |
| 1500 ms | -7.73 | [-31.85, +16.39] | .766 | 0-2 |
| 3000 ms | +3.12 | [-36.58, +42.83] | .842 | 1-0 |
| 4000 ms | +5.63 | [-49.96, +61.21] | .842 | 2-0 |
| Avg | -1.05 | [-4.94, +2.85] | .766 | 0-2 |

- Window average: EEGNet 50 Hz ahead in 1 of 3 pairs; largest lead HOC/WFE (+1.94), largest deficit EPS/SPS (-3.19)
- Pearson r between per-pair window averages: 0.51 (p .659)

## EEGNet 50 Hz minus ShallowConvNet, paired over subjects (pooled over 3 pairs)

| Window | Diff | 95% CI | Holm p | Subjects EEGNet 50 Hz better / worse |
|---|---:|---:|---:|---:|
| 800 ms | +3.94 | [+0.99, +6.88] | .187 | 2-0 |
| 1500 ms | +1.78 | [-44.40, +47.96] | .710 | 1-1 |
| 3000 ms | +5.65 | [-28.47, +39.77] | .577 | 2-0 |
| 4000 ms | +8.01 | [+1.54, +14.48] | .187 | 2-0 |
| Avg | +4.84 | [-14.35, +24.04] | .577 | 2-0 |

- Window average: EEGNet 50 Hz ahead in 3 of 3 pairs; largest lead EPS/SPS (+5.54), largest deficit WAA/SAA (+4.06)
- Pearson r between per-pair window averages: 1.00 (p .024)

## Per pair (sorted by the first EEGNet model's window average)

| Pair | Group | EEGNet 250 Hz 800 ms | EEGNet 250 Hz 1500 ms | EEGNet 250 Hz 3000 ms | EEGNet 250 Hz 4000 ms | EEGNet 250 Hz Avg | EEGNet 50 Hz 800 ms | EEGNet 50 Hz 1500 ms | EEGNet 50 Hz 3000 ms | EEGNet 50 Hz 4000 ms | EEGNet 50 Hz Avg | LEAN Avg | EEG Conformer Avg | ShallowConvNet Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WAA/SAA | Distant | 53.06 | 62.43 | 61.32 | 58.33 | 58.78 | 54.31 | 46.46 | 53.47 | 63.26 | 54.38 | 46.82 | 56.27 | 50.31 |
| EPS/SPS | Adjacent | 59.31 | 56.88 | 53.33 | 58.33 | 56.96 | 58.26 | 52.15 | 57.85 | 59.24 | 56.88 | 52.59 | 60.07 | 51.34 |
| HOC/WFE | Adjacent | 58.47 | 51.11 | 50.69 | 56.94 | 54.31 | 52.08 | 54.24 | 61.18 | 56.11 | 55.90 | 49.88 | 53.96 | 50.97 |

Note: smoke-mode numbers only prove the pipeline runs; they say nothing about accuracy.

