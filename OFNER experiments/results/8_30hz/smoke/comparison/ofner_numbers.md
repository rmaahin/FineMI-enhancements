# LEAN vs EEG Conformer on OFNER, 8-30 Hz (smoke mode, 3 pairs, 2 subjects)

Data: Ofner et al. 2017, motor imagery, 0 to 3 s after the cue, fixation-cross baseline (-2 to 0 s), 8-30 Hz, kept at 250 Hz (no decimation). Windows start at the cue.

## Parameters

| Model | 800 ms | 1500 ms | 3000 ms |
|---|---:|---:|---:|
| LEAN | 38,071 | 38,071 | 38,071 |
| EEG Conformer | 151,738 | 167,098 | 199,098 |

## Mean accuracy over the selected pairs

| Model | 800 ms | 1500 ms | 3000 ms | Avg |
|---|---:|---:|---:|---:|
| LEAN | 50.28 | 51.25 | 51.53 | 51.02 |
| EEG Conformer | 48.61 | 54.31 | 48.19 | 50.37 |

## LEAN minus EEG Conformer, paired over subjects (pooled over 3 pairs)

| Window | Diff | 95% CI | Holm p | Subjects LEAN better / Conformer better |
|---|---:|---:|---:|---:|
| 800 ms | +1.67 | [-44.22, +47.55] | 1.000 | 1-1 |
| 1500 ms | -3.06 | [-24.23, +18.12] | .954 | 0-2 |
| 3000 ms | +3.33 | [-7.26, +13.92] | .624 | 2-0 |
| Avg | +0.65 | [-11.12, +12.41] | 1.000 | 1-1 |

## Window average (800 / 1500 / 3000 ms)

- LEAN overall average: **51.02%**
- EEG Conformer overall average: **50.37%**
- Average gap (Conformer - LEAN): **-0.65** points
- LEAN retains **101.3%** of EEG Conformer's average accuracy
- Gap below 4 points in **3 of 3** pairs; LEAN ahead in 2 pair(s)
- Three smallest gaps: HC/HO (-3.06), SUP/PRO (-0.42), EF/EE (1.53)
- Pearson r between the two models' per-pair accuracies: **0.98** (p .111)

| Group | Pairs | LEAN | EEG Conformer | Gap |
|---|---:|---:|---:|---:|
| Same-joint | 3 | 51.02 | 50.37 | -0.65 |
| Overall | 3 | 51.02 | 50.37 | -0.65 |

## 3000 ms window only

- LEAN overall average: **51.53%**
- EEG Conformer overall average: **48.19%**
- Average gap (Conformer - LEAN): **-3.33** points
- LEAN retains **106.9%** of EEG Conformer's average accuracy
- Gap below 4 points in **3 of 3** pairs; LEAN ahead in 3 pair(s)
- Three smallest gaps: HC/HO (-8.33), SUP/PRO (-1.25), EF/EE (-0.42)
- Pearson r between the two models' per-pair accuracies: **0.23** (p .849)

| Group | Pairs | LEAN | EEG Conformer | Gap |
|---|---:|---:|---:|---:|
| Same-joint | 3 | 51.53 | 48.19 | -3.33 |
| Overall | 3 | 51.53 | 48.19 | -3.33 |

## Table rows, window average (EEG Conformer & LEAN; best in bold)

```
EF/EE & \textbf{49.86} & 48.33 \\
SUP/PRO & 50.14 & \textbf{50.56} \\
HC/HO & 51.11 & \textbf{54.17} \\
Same-joint avg. & 50.37 & 51.02 \\
Overall avg. & 50.37 & 51.02 \\
```

## Table rows, 3000 ms (EEG Conformer & LEAN; best in bold)

```
EF/EE & 48.75 & \textbf{49.17} \\
SUP/PRO & 47.50 & \textbf{48.75} \\
HC/HO & 48.33 & \textbf{56.67} \\
Same-joint avg. & 48.19 & 51.53 \\
Overall avg. & 48.19 & 51.53 \\
```

Note: smoke-mode numbers only prove the pipeline runs; they say nothing about accuracy.

