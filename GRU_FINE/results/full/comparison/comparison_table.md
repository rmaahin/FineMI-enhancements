# FineMI 0.5-3 Hz, decimated to 50 Hz (D=5): architecture comparison (full mode)

Within-subject CV test accuracy (%), mean (SD) across subjects. Bold = best architecture for that pair and window.

## Pooled over 3 pairs

Each subject's accuracy averaged over the pairs first; 'pairs best' = number of pairs on which the architecture has the highest mean (ties count for each tied architecture).

| Architecture | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| FINE + GRU | **65.25 (7.99)** | **68.19 (8.36)** | **65.05 (8.23)** | **64.71 (7.94)** | **65.80 (7.49)** |
| FINE + GRU + CWT | 59.93 (8.51) | 61.59 (7.69) | 59.18 (4.76) | 58.53 (6.41) | 59.81 (5.64) |

| Pairs best | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| FINE + GRU | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| FINE + GRU + CWT | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |

## Per pair

| Pair | Architecture | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---|---:|---:|---:|---:|---:|
| WAA vs SAA | FINE + GRU | **65.93 (11.65)** | **71.57 (12.81)** | **69.24 (12.38)** | **69.43 (11.44)** | **69.04 (10.56)** |
|  | FINE + GRU + CWT | 59.68 (9.64) | 61.93 (11.33) | 60.63 (8.23) | 60.03 (7.94) | 60.57 (7.35) |
| HOC vs WFE | FINE + GRU | **62.89 (7.06)** | **62.15 (9.27)** | **59.31 (8.61)** | **60.29 (10.68)** | **61.16 (8.10)** |
|  | FINE + GRU + CWT | 58.58 (8.85) | 58.32 (6.45) | 57.41 (4.76) | 57.01 (6.55) | 57.83 (4.51) |
| EPS vs SPS | FINE + GRU | **66.95 (11.54)** | **70.86 (11.70)** | **66.60 (10.66)** | **64.40 (10.32)** | **67.20 (9.66)** |
|  | FINE + GRU + CWT | 61.53 (13.80) | 64.54 (10.69) | 59.51 (9.43) | 58.56 (10.05) | 61.03 (8.78) |

Subjects per cell: [18]
