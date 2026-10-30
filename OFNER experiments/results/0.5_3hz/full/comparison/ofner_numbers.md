# LEAN vs EEG Conformer on OFNER, 0.5-3 Hz (full mode, 15 pairs, 15 subjects)

Data: Ofner et al. 2017, motor imagery, 0 to 3 s after the cue, fixation-cross baseline (-2 to 0 s), 0.5-3 Hz, decimated to 50 Hz. Windows start at the cue.

## Parameters

| Model | 800 ms | 1500 ms | 3000 ms |
|---|---:|---:|---:|
| LEAN | 25,139 | 25,139 | 25,139 |
| EEG Conformer | 151,738 | 167,098 | 199,098 |

## Mean accuracy over the selected pairs

| Model | 800 ms | 1500 ms | 3000 ms | Avg |
|---|---:|---:|---:|---:|
| LEAN | 50.19 | 52.07 | 50.94 | 51.07 |
| EEG Conformer | 50.53 | 52.61 | 52.26 | 51.80 |

## LEAN minus EEG Conformer, paired over subjects (pooled over 15 pairs)

| Window | Diff | 95% CI | Holm p | Subjects LEAN better / Conformer better |
|---|---:|---:|---:|---:|
| 800 ms | -0.34 | [-1.18, +0.49] | .496 | 5-9 |
| 1500 ms | -0.54 | [-1.50, +0.42] | .496 | 6-9 |
| 3000 ms | -1.31 | [-2.17, -0.46] | .022 | 3-12 |
| Avg | -0.73 | [-1.25, -0.21] | .028 | 3-12 |

## Window average (800 / 1500 / 3000 ms)

- LEAN overall average: **51.07%**
- EEG Conformer overall average: **51.80%**
- Average gap (Conformer - LEAN): **0.73** points
- LEAN retains **98.6%** of EEG Conformer's average accuracy
- Gap below 4 points in **15 of 15** pairs; LEAN ahead in 4 pair(s)
- Three smallest gaps: EF/EE (-1.09), EF/PRO (-0.46), EF/SUP (-0.19)
- Pearson r between the two models' per-pair accuracies: **0.40** (p .141)

| Group | Pairs | LEAN | EEG Conformer | Gap |
|---|---:|---:|---:|---:|
| Cross-joint | 12 | 51.08 | 52.03 | 0.96 |
| Same-joint | 3 | 51.03 | 50.86 | -0.17 |
| Overall | 15 | 51.07 | 51.80 | 0.73 |

## 3000 ms window only

- LEAN overall average: **50.94%**
- EEG Conformer overall average: **52.26%**
- Average gap (Conformer - LEAN): **1.31** points
- LEAN retains **97.5%** of EEG Conformer's average accuracy
- Gap below 4 points in **15 of 15** pairs; LEAN ahead in 4 pair(s)
- Three smallest gaps: EF/EE (-2.39), EF/PRO (-0.44), PRO/HO (-0.33)
- Pearson r between the two models' per-pair accuracies: **0.20** (p .476)

| Group | Pairs | LEAN | EEG Conformer | Gap |
|---|---:|---:|---:|---:|
| Cross-joint | 12 | 50.88 | 52.61 | 1.73 |
| Same-joint | 3 | 51.20 | 50.85 | -0.35 |
| Overall | 15 | 50.94 | 52.26 | 1.31 |

## Table rows, window average (EEG Conformer & LEAN; best in bold)

```
EF/EE & 50.74 & \textbf{51.83} \\
EF/SUP & 50.57 & \textbf{50.76} \\
EF/PRO & 50.87 & \textbf{51.33} \\
EF/HC & \textbf{52.98} & 51.04 \\
EF/HO & \textbf{52.26} & 51.44 \\
EE/SUP & \textbf{51.83} & 49.91 \\
EE/PRO & \textbf{53.24} & 51.39 \\
EE/HC & \textbf{52.74} & 52.44 \\
EE/HO & \textbf{52.15} & 50.28 \\
SUP/PRO & 51.26 & \textbf{51.39} \\
SUP/HC & \textbf{50.65} & 50.09 \\
SUP/HO & \textbf{53.19} & 51.67 \\
PRO/HC & \textbf{52.57} & 51.24 \\
PRO/HO & \textbf{51.35} & 51.31 \\
HC/HO & \textbf{50.57} & 49.87 \\
Cross-joint avg. & 52.03 & 51.08 \\
Same-joint avg. & 50.86 & 51.03 \\
Overall avg. & 51.80 & 51.07 \\
```

## Table rows, 3000 ms (EEG Conformer & LEAN; best in bold)

```
EF/EE & 50.28 & \textbf{52.67} \\
EF/SUP & \textbf{51.11} & 50.56 \\
EF/PRO & 50.33 & \textbf{50.78} \\
EF/HC & \textbf{54.39} & 52.17 \\
EF/HO & \textbf{52.06} & 50.28 \\
EE/SUP & \textbf{52.33} & 50.44 \\
EE/PRO & \textbf{54.06} & 52.06 \\
EE/HC & \textbf{54.11} & 52.22 \\
EE/HO & \textbf{52.61} & 50.28 \\
SUP/PRO & 51.06 & \textbf{51.28} \\
SUP/HC & \textbf{52.28} & 49.11 \\
SUP/HO & \textbf{54.11} & 50.44 \\
PRO/HC & \textbf{53.11} & 51.11 \\
PRO/HO & 50.78 & \textbf{51.11} \\
HC/HO & \textbf{51.22} & 49.67 \\
Cross-joint avg. & 52.61 & 50.88 \\
Same-joint avg. & 50.85 & 51.20 \\
Overall avg. & 52.26 & 50.94 \\
```

