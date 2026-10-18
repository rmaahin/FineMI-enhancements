# LEAN vs EEG Conformer (smoke mode, 3 pairs, 2 subjects)

Data: fixation-cross baseline (-4 to -2 s), motor imagery only (0 to 4 s after the trigger), 0.5-3 Hz, decimated to 50 Hz. Windows start at imagery onset.

## Parameters

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms |
|---|---:|---:|---:|---:|
| LEAN | 25,752 | 25,752 | 25,752 | 25,752 |
| EEG Conformer | 153,978 | 169,338 | 201,338 | 221,818 |

## Mean accuracy over the selected pairs

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| LEAN | 52.06 | 50.49 | 46.90 | 49.61 | 49.76 |
| EEG Conformer | 60.09 | 58.68 | 54.38 | 53.91 | 56.77 |

## LEAN minus EEG Conformer, paired over subjects (pooled over 3 pairs)

| Window | Diff | 95% CI | Holm p | Subjects LEAN better / Conformer better |
|---|---:|---:|---:|---:|
| 800 ms | -8.03 | [-48.92, +32.85] | .793 | 0-2 |
| 1500 ms | -8.19 | [-41.72, +25.34] | .793 | 0-2 |
| 3000 ms | -7.48 | [-13.07, -1.89] | .187 | 0-2 |
| 4000 ms | -4.31 | [-44.89, +36.28] | .793 | 0-2 |
| Avg | -7.00 | [-37.15, +23.15] | .793 | 0-2 |

## Paper numbers: window average (as in the September paper)

- LEAN overall average: **49.76%**
- EEG Conformer overall average: **56.77%**
- Average gap (Conformer - LEAN): **7.00** points
- LEAN retains **87.7%** of EEG Conformer's average accuracy
- Gap below 4 points in **0 of 3** pairs; LEAN ahead in 0 pair(s)
- Three smallest gaps: HOC/WFE (4.08), EPS/SPS (7.48), WAA/SAA (9.44)
- Pearson r between the two models' per-pair accuracies: **0.59** (p .600)

| Group | Pairs | LEAN | EEG Conformer | Gap |
|---|---:|---:|---:|---:|
| Distant | 1 | 46.82 | 56.27 | 9.44 |
| Adjacent | 2 | 51.23 | 57.01 | 5.78 |
| Overall | 3 | 49.76 | 56.77 | 7.00 |

## Paper numbers: 4000 ms window only

- LEAN overall average: **49.61%**
- EEG Conformer overall average: **53.91%**
- Average gap (Conformer - LEAN): **4.31** points
- LEAN retains **92.0%** of EEG Conformer's average accuracy
- Gap below 4 points in **1 of 3** pairs; LEAN ahead in 0 pair(s)
- Three smallest gaps: EPS/SPS (2.08), HOC/WFE (4.31), WAA/SAA (6.53)
- Pearson r between the two models' per-pair accuracies: **0.70** (p .503)

| Group | Pairs | LEAN | EEG Conformer | Gap |
|---|---:|---:|---:|---:|
| Distant | 1 | 50.07 | 56.60 | 6.53 |
| Adjacent | 2 | 49.38 | 52.57 | 3.19 |
| Overall | 3 | 49.61 | 53.91 | 4.31 |

## Table 2 rows, window average (EEG Conformer & LEAN; best in bold)

```
WAA/SAA & \textbf{56.27} & 46.82 \\
HOC/WFE & \textbf{53.96} & 49.88 \\
EPS/SPS & \textbf{60.07} & 52.59 \\
Distant avg. & 56.27 & 46.82 \\
Adjacent avg. & 57.01 & 51.23 \\
Overall avg. & 56.77 & 49.76 \\
```

## Table 2 rows, 4000 ms (EEG Conformer & LEAN; best in bold)

```
WAA/SAA & \textbf{56.60} & 50.07 \\
HOC/WFE & \textbf{50.83} & 46.53 \\
EPS/SPS & \textbf{54.31} & 52.22 \\
Distant avg. & 56.60 & 50.07 \\
Adjacent avg. & 52.57 & 49.38 \\
Overall avg. & 53.91 & 49.61 \\
```

Note: smoke-mode numbers only prove the pipeline runs; they say nothing about accuracy.

