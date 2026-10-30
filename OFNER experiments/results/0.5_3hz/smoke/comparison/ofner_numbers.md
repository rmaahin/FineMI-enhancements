# LEAN vs EEG Conformer on OFNER, 0.5-3 Hz (smoke mode, 3 pairs, 2 subjects)

Data: Ofner et al. 2017, motor imagery, 0 to 3 s after the cue, fixation-cross baseline (-2 to 0 s), 0.5-3 Hz, decimated to 50 Hz. Windows start at the cue.

## Parameters

| Model | 800 ms | 1500 ms | 3000 ms |
|---|---:|---:|---:|
| LEAN | 25,139 | 25,139 | 25,139 |
| EEG Conformer | 151,738 | 167,098 | 199,098 |

## Mean accuracy over the selected pairs

| Model | 800 ms | 1500 ms | 3000 ms | Avg |
|---|---:|---:|---:|---:|
| LEAN | 51.94 | 49.44 | 47.36 | 49.58 |
| EEG Conformer | 51.94 | 52.22 | 50.42 | 51.53 |

## LEAN minus EEG Conformer, paired over subjects (pooled over 3 pairs)

| Window | Diff | 95% CI | Holm p | Subjects LEAN better / Conformer better |
|---|---:|---:|---:|---:|
| 800 ms | -0.00 | [-7.06, +7.06] | 1.000 | 1-1 |
| 1500 ms | -2.78 | [-34.54, +28.99] | 1.000 | 0-2 |
| 3000 ms | -3.06 | [-38.35, +32.24] | 1.000 | 0-2 |
| Avg | -1.94 | [-21.94, +18.06] | 1.000 | 0-2 |

## Window average (800 / 1500 / 3000 ms)

- LEAN overall average: **49.58%**
- EEG Conformer overall average: **51.53%**
- Average gap (Conformer - LEAN): **1.94** points
- LEAN retains **96.2%** of EEG Conformer's average accuracy
- Gap below 4 points in **2 of 3** pairs; LEAN ahead in 1 pair(s)
- Three smallest gaps: EF/EE (-0.14), HC/HO (1.53), SUP/PRO (4.44)
- Pearson r between the two models' per-pair accuracies: **-0.77** (p .436)

| Group | Pairs | LEAN | EEG Conformer | Gap |
|---|---:|---:|---:|---:|
| Same-joint | 3 | 49.58 | 51.53 | 1.94 |
| Overall | 3 | 49.58 | 51.53 | 1.94 |

## 3000 ms window only

- LEAN overall average: **47.36%**
- EEG Conformer overall average: **50.42%**
- Average gap (Conformer - LEAN): **3.06** points
- LEAN retains **93.9%** of EEG Conformer's average accuracy
- Gap below 4 points in **2 of 3** pairs; LEAN ahead in 0 pair(s)
- Three smallest gaps: SUP/PRO (1.67), HC/HO (2.50), EF/EE (5.00)
- Pearson r between the two models' per-pair accuracies: **0.72** (p .488)

| Group | Pairs | LEAN | EEG Conformer | Gap |
|---|---:|---:|---:|---:|
| Same-joint | 3 | 47.36 | 50.42 | 3.06 |
| Overall | 3 | 47.36 | 50.42 | 3.06 |

## Table rows, window average (EEG Conformer & LEAN; best in bold)

```
EF/EE & 50.28 & \textbf{50.42} \\
SUP/PRO & \textbf{52.50} & 48.06 \\
HC/HO & \textbf{51.81} & 50.28 \\
Same-joint avg. & 51.53 & 49.58 \\
Overall avg. & 51.53 & 49.58 \\
```

## Table rows, 3000 ms (EEG Conformer & LEAN; best in bold)

```
EF/EE & \textbf{52.92} & 47.92 \\
SUP/PRO & \textbf{50.42} & 48.75 \\
HC/HO & \textbf{47.92} & 45.42 \\
Same-joint avg. & 50.42 & 47.36 \\
Overall avg. & 50.42 & 47.36 \\
```

Note: smoke-mode numbers only prove the pipeline runs; they say nothing about accuracy.

