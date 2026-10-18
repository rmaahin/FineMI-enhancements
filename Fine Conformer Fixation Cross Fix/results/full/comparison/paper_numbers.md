# LEAN vs EEG Conformer (full mode, 28 pairs, 18 subjects)

Data: fixation-cross baseline (-4 to -2 s), motor imagery only (0 to 4 s after the trigger), 0.5-3 Hz, decimated to 50 Hz. Windows start at imagery onset.

## Parameters

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms |
|---|---:|---:|---:|---:|
| LEAN | 25,752 | 25,752 | 25,752 | 25,752 |
| EEG Conformer | 153,978 | 169,338 | 201,338 | 221,818 |

## Mean accuracy over the selected pairs

| Model | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| LEAN | 60.06 | 58.29 | 55.67 | 55.17 | 57.30 |
| EEG Conformer | 62.19 | 61.59 | 60.99 | 60.86 | 61.41 |

## LEAN minus EEG Conformer, paired over subjects (pooled over 28 pairs)

| Window | Diff | 95% CI | Holm p | Subjects LEAN better / Conformer better |
|---|---:|---:|---:|---:|
| 800 ms | -2.13 | [-3.01, -1.25] | <.001 | 2-16 |
| 1500 ms | -3.30 | [-4.53, -2.07] | <.001 | 1-17 |
| 3000 ms | -5.32 | [-7.97, -2.66] | <.001 | 2-16 |
| 4000 ms | -5.69 | [-7.95, -3.44] | <.001 | 1-17 |
| Avg | -4.11 | [-5.66, -2.56] | <.001 | 1-17 |

## Paper numbers: window average (as in the September paper)

- LEAN overall average: **57.30%**
- EEG Conformer overall average: **61.41%**
- Average gap (Conformer - LEAN): **4.11** points
- LEAN retains **93.3%** of EEG Conformer's average accuracy
- Gap below 4 points in **14 of 28** pairs; LEAN ahead in 1 pair(s)
- Three smallest gaps: WFE/WAA (-0.19), HOC/EPS (1.27), HOC/WAA (1.28)
- Ten pairs where EEG Conformer is lowest: average gap **2.21**; its ten best pairs: **6.04**
- Pearson r between the two models' per-pair accuracies: **0.84** (p <.001)

| Group | Pairs | LEAN | EEG Conformer | Gap |
|---|---:|---:|---:|---:|
| Distant | 11 | 58.59 | 62.73 | 4.14 |
| Adjacent | 12 | 56.29 | 60.26 | 3.97 |
| Same-joint | 5 | 56.86 | 61.23 | 4.38 |
| Overall | 28 | 57.30 | 61.41 | 4.11 |

## Paper numbers: 4000 ms window only

- LEAN overall average: **55.17%**
- EEG Conformer overall average: **60.86%**
- Average gap (Conformer - LEAN): **5.69** points
- LEAN retains **90.6%** of EEG Conformer's average accuracy
- Gap below 4 points in **9 of 28** pairs; LEAN ahead in 1 pair(s)
- Three smallest gaps: WFE/WAA (-0.61), WAA/EFE (0.88), EPS/EFE (1.60)
- Ten pairs where EEG Conformer is lowest: average gap **3.26**; its ten best pairs: **8.48**
- Pearson r between the two models' per-pair accuracies: **0.38** (p .045)

| Group | Pairs | LEAN | EEG Conformer | Gap |
|---|---:|---:|---:|---:|
| Distant | 11 | 56.02 | 61.82 | 5.80 |
| Adjacent | 12 | 54.47 | 60.16 | 5.70 |
| Same-joint | 5 | 54.96 | 60.41 | 5.44 |
| Overall | 28 | 55.17 | 60.86 | 5.69 |

## Table 2 rows, window average (EEG Conformer & LEAN; best in bold)

```
HOC/WFE & \textbf{56.86} & 54.26 \\
HOC/WAA & \textbf{58.78} & 57.50 \\
HOC/EPS & \textbf{57.12} & 55.85 \\
HOC/EFE & \textbf{57.94} & 55.55 \\
HOC/SPS & \textbf{65.27} & 59.68 \\
HOC/SAA & \textbf{66.49} & 62.68 \\
HOC/SFE & \textbf{60.79} & 57.18 \\
WFE/WAA & 57.06 & \textbf{57.25} \\
WFE/EPS & \textbf{57.16} & 54.52 \\
WFE/EFE & \textbf{56.85} & 53.04 \\
WFE/SPS & \textbf{64.34} & 58.66 \\
WFE/SAA & \textbf{64.23} & 59.67 \\
WFE/SFE & \textbf{59.60} & 57.45 \\
WAA/EPS & \textbf{57.05} & 53.37 \\
WAA/EFE & \textbf{58.03} & 55.57 \\
WAA/SPS & \textbf{65.93} & 60.18 \\
WAA/SAA & \textbf{66.27} & 59.94 \\
WAA/SFE & \textbf{62.09} & 57.72 \\
EPS/EFE & \textbf{57.71} & 55.50 \\
EPS/SPS & \textbf{67.34} & 58.83 \\
EPS/SAA & \textbf{64.94} & 59.95 \\
EPS/SFE & \textbf{64.05} & 58.42 \\
EFE/SPS & \textbf{60.78} & 55.91 \\
EFE/SAA & \textbf{61.22} & 58.37 \\
EFE/SFE & \textbf{60.11} & 55.79 \\
SPS/SAA & \textbf{62.29} & 57.55 \\
SPS/SFE & \textbf{64.73} & 58.28 \\
SAA/SFE & \textbf{64.39} & 55.68 \\
Distant avg. & 62.73 & 58.59 \\
Adjacent avg. & 60.26 & 56.29 \\
Same-joint avg. & 61.23 & 56.86 \\
Overall avg. & 61.41 & 57.30 \\
```

## Table 2 rows, 4000 ms (EEG Conformer & LEAN; best in bold)

```
HOC/WFE & \textbf{56.90} & 52.99 \\
HOC/WAA & \textbf{58.76} & 56.75 \\
HOC/EPS & \textbf{58.70} & 54.24 \\
HOC/EFE & \textbf{58.08} & 54.01 \\
HOC/SPS & \textbf{62.77} & 55.70 \\
HOC/SAA & \textbf{65.49} & 59.98 \\
HOC/SFE & \textbf{59.58} & 54.91 \\
WFE/WAA & 56.27 & \textbf{56.88} \\
WFE/EPS & \textbf{58.22} & 51.84 \\
WFE/EFE & \textbf{56.31} & 52.51 \\
WFE/SPS & \textbf{62.48} & 54.33 \\
WFE/SAA & \textbf{61.25} & 57.49 \\
WFE/SFE & \textbf{59.26} & 56.68 \\
WAA/EPS & \textbf{59.72} & 52.85 \\
WAA/EFE & \textbf{56.71} & 55.83 \\
WAA/SPS & \textbf{65.13} & 55.74 \\
WAA/SAA & \textbf{64.41} & 55.76 \\
WAA/SFE & \textbf{62.91} & 57.39 \\
EPS/EFE & \textbf{56.89} & 55.29 \\
EPS/SPS & \textbf{67.11} & 55.76 \\
EPS/SAA & \textbf{65.03} & 56.88 \\
EPS/SFE & \textbf{64.29} & 54.55 \\
EFE/SPS & \textbf{59.49} & 53.16 \\
EFE/SAA & \textbf{60.64} & 57.78 \\
EFE/SFE & \textbf{58.80} & 52.70 \\
SPS/SAA & \textbf{60.46} & 53.68 \\
SPS/SFE & \textbf{64.34} & 56.34 \\
SAA/SFE & \textbf{64.07} & 52.62 \\
Distant avg. & 61.82 & 56.02 \\
Adjacent avg. & 60.16 & 54.47 \\
Same-joint avg. & 60.41 & 54.96 \\
Overall avg. & 60.86 & 55.17 \\
```

