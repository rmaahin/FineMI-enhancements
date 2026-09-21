# FineMI 0.5-3 Hz, decimated to 50 Hz (D=5): architecture comparison (smoke mode)

Within-subject CV test accuracy (%), mean (SD) across subjects. Bold = best architecture for that pair and window.

## Pooled over 3 pairs

Each subject's accuracy averaged over the pairs first; 'pairs best' = number of pairs on which the architecture has the highest mean (ties count for each tied architecture).

| Architecture | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| Conformer A | **54.63 (7.59)** | **61.60 (0.69)** | **56.92 (6.12)** | **57.96 (12.31)** | **57.78 (6.68)** |
| Conformer B | 52.15 (0.10) | 51.55 (4.29) | 50.21 (5.01) | 50.69 (11.98) | 51.15 (5.30) |
| Conformer B + CWT | 50.97 (0.20) | 49.61 (0.03) | 51.85 (2.10) | 52.62 (1.60) | 51.26 (0.87) |

| Pairs best | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| Conformer A | 2/3 | 3/3 | 2/3 | 2/3 | 3/3 |
| Conformer B | 1/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| Conformer B + CWT | 0/3 | 0/3 | 1/3 | 1/3 | 0/3 |

## Per pair

| Pair | Architecture | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---|---:|---:|---:|---:|---:|
| HOC vs WFE | Conformer A | **54.51 (9.53)** | **56.39 (1.96)** | 48.19 (0.98) | 51.53 (8.45) | **52.66 (3.76)** |
|  | Conformer B | 50.56 (0.79) | 49.58 (4.12) | 47.15 (2.26) | 43.54 (14.44) | 47.71 (5.01) |
|  | Conformer B + CWT | 48.40 (4.03) | 46.94 (2.75) | **52.57 (5.21)** | **52.64 (6.87)** | 50.14 (3.34) |
| WAA vs SAA | Conformer A | **59.72 (7.46)** | **62.01 (5.99)** | **63.82 (6.97)** | **61.81 (15.12)** | **61.84 (8.89)** |
|  | Conformer B | 55.76 (2.85) | 54.93 (5.40) | 50.90 (7.56) | 55.21 (12.08) | 54.20 (5.55) |
|  | Conformer B + CWT | 55.90 (0.49) | 47.01 (1.08) | 48.26 (0.69) | 53.54 (0.29) | 51.18 (0.10) |
| EPS vs SPS | Conformer A | 49.65 (5.79) | **66.39 (1.96)** | **58.75 (12.37)** | **60.56 (13.36)** | **58.84 (7.39)** |
|  | Conformer B | **50.14 (3.34)** | 50.14 (3.34) | 52.57 (5.21) | 53.33 (9.43) | 51.55 (5.33) |
|  | Conformer B + CWT | 48.61 (5.11) | 54.86 (3.73) | 54.72 (0.39) | 51.67 (2.36) | 52.47 (0.83) |

Subjects per cell: [2]
