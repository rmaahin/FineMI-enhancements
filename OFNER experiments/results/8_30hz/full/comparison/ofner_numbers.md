# LEAN vs EEG Conformer on OFNER, 8-30 Hz (full mode, 15 pairs, 15 subjects)

Data: Ofner et al. 2017, motor imagery, 0 to 3 s after the cue, fixation-cross baseline (-2 to 0 s), 8-30 Hz, kept at 250 Hz (no decimation). Windows start at the cue.

## Parameters

| Model | 800 ms | 1500 ms | 3000 ms |
|---|---:|---:|---:|
| LEAN | 38,071 | 38,071 | 38,071 |
| EEG Conformer | 151,738 | 167,098 | 199,098 |

## Mean accuracy over the selected pairs

| Model | 800 ms | 1500 ms | 3000 ms | Avg |
|---|---:|---:|---:|---:|
| LEAN | 49.94 | 50.24 | 52.83 | 51.00 |
| EEG Conformer | 50.34 | 50.76 | 52.24 | 51.11 |

## LEAN minus EEG Conformer, paired over subjects (pooled over 15 pairs)

| Window | Diff | 95% CI | Holm p | Subjects LEAN better / Conformer better |
|---|---:|---:|---:|---:|
| 800 ms | -0.40 | [-1.36, +0.57] | .789 | 5-10 |
| 1500 ms | -0.52 | [-1.27, +0.24] | .649 | 6-9 |
| 3000 ms | +0.59 | [-0.47, +1.64] | .756 | 9-6 |
| Avg | -0.11 | [-0.73, +0.51] | .789 | 8-7 |

## Window average (800 / 1500 / 3000 ms)

- LEAN overall average: **51.00%**
- EEG Conformer overall average: **51.11%**
- Average gap (Conformer - LEAN): **0.11** points
- LEAN retains **99.8%** of EEG Conformer's average accuracy
- Gap below 4 points in **15 of 15** pairs; LEAN ahead in 7 pair(s)
- Three smallest gaps: SUP/PRO (-1.06), SUP/HO (-0.93), EF/EE (-0.89)
- Pearson r between the two models' per-pair accuracies: **0.43** (p .113)

| Group | Pairs | LEAN | EEG Conformer | Gap |
|---|---:|---:|---:|---:|
| Cross-joint | 12 | 51.04 | 51.39 | 0.35 |
| Same-joint | 3 | 50.85 | 49.99 | -0.85 |
| Overall | 15 | 51.00 | 51.11 | 0.11 |

## 3000 ms window only

- LEAN overall average: **52.83%**
- EEG Conformer overall average: **52.24%**
- Average gap (Conformer - LEAN): **-0.59** points
- LEAN retains **101.1%** of EEG Conformer's average accuracy
- Gap below 4 points in **15 of 15** pairs; LEAN ahead in 12 pair(s)
- Three smallest gaps: EE/HC (-2.28), SUP/PRO (-2.17), SUP/HC (-1.33)
- Pearson r between the two models' per-pair accuracies: **0.82** (p <.001)

| Group | Pairs | LEAN | EEG Conformer | Gap |
|---|---:|---:|---:|---:|
| Cross-joint | 12 | 53.21 | 52.73 | -0.48 |
| Same-joint | 3 | 51.30 | 50.28 | -1.02 |
| Overall | 15 | 52.83 | 52.24 | -0.59 |

## Table rows, window average (EEG Conformer & LEAN; best in bold)

```
EF/EE & 50.13 & \textbf{51.02} \\
EF/SUP & \textbf{50.87} & 50.65 \\
EF/PRO & 51.06 & \textbf{51.89} \\
EF/HC & \textbf{52.44} & 50.74 \\
EF/HO & 51.07 & \textbf{51.31} \\
EE/SUP & \textbf{50.35} & 50.04 \\
EE/PRO & \textbf{52.26} & 51.74 \\
EE/HC & \textbf{52.72} & 52.65 \\
EE/HO & \textbf{52.54} & 49.94 \\
SUP/PRO & 49.48 & \textbf{50.54} \\
SUP/HC & \textbf{51.37} & 51.11 \\
SUP/HO & 50.00 & \textbf{50.93} \\
PRO/HC & 51.78 & \textbf{51.81} \\
PRO/HO & \textbf{50.24} & 49.69 \\
HC/HO & 50.37 & \textbf{50.98} \\
Cross-joint avg. & 51.39 & 51.04 \\
Same-joint avg. & 49.99 & 50.85 \\
Overall avg. & 51.11 & 51.00 \\
```

## Table rows, 3000 ms (EEG Conformer & LEAN; best in bold)

```
EF/EE & 50.61 & \textbf{51.11} \\
EF/SUP & 50.67 & \textbf{51.06} \\
EF/PRO & 52.06 & \textbf{52.39} \\
EF/HC & \textbf{54.11} & 52.22 \\
EF/HO & 52.67 & \textbf{53.89} \\
EE/SUP & 51.44 & \textbf{52.44} \\
EE/PRO & 52.50 & \textbf{52.67} \\
EE/HC & 54.94 & \textbf{57.22} \\
EE/HO & 53.28 & \textbf{53.50} \\
SUP/PRO & 49.28 & \textbf{51.44} \\
SUP/HC & 53.83 & \textbf{55.17} \\
SUP/HO & 51.67 & \textbf{52.89} \\
PRO/HC & \textbf{54.50} & 54.28 \\
PRO/HO & \textbf{51.11} & 50.78 \\
HC/HO & 50.94 & \textbf{51.33} \\
Cross-joint avg. & 52.73 & 53.21 \\
Same-joint avg. & 50.28 & 51.30 \\
Overall avg. & 52.24 & 52.83 \\
```

