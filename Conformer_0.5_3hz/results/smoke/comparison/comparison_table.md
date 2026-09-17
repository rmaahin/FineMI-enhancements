# FineMI 0.5-3 Hz: architecture comparison (smoke mode)

Within-subject CV test accuracy (%), mean (SD) across subjects. Bold = best architecture for that pair and window.

## Pooled over 28 pairs

Each subject's accuracy averaged over the pairs first; 'pairs best' = number of pairs on which the architecture has the highest mean (ties count for each tied architecture).

| Architecture | 800 ms | Avg |
|---|---:|---:|
| FINE (base) | 53.57 | 53.57 |
| Conformer A | 53.57 | 53.57 |
| Conformer B | **53.77** | **53.77** |

| Pairs best | 800 ms | Avg |
|---|---:|---:|
| FINE (base) | 12/28 | 12/28 |
| Conformer A | 13/28 | 13/28 |
| Conformer B | 10/28 | 10/28 |

## Per pair

| Pair | Architecture | 800 ms | Avg |
|---|---|---:|---:|
| HOC vs WFE | FINE (base) | **50.00** | **50.00** |
|  | Conformer A | 38.89 | 38.89 |
|  | Conformer B | 44.44 | 44.44 |
| HOC vs WAA | FINE (base) | 50.00 | 50.00 |
|  | Conformer A | **61.11** | **61.11** |
|  | Conformer B | 38.89 | 38.89 |
| HOC vs EPS | FINE (base) | 33.33 | 33.33 |
|  | Conformer A | 44.44 | 44.44 |
|  | Conformer B | **66.67** | **66.67** |
| HOC vs EFE | FINE (base) | 50.00 | 50.00 |
|  | Conformer A | **72.22** | **72.22** |
|  | Conformer B | **72.22** | **72.22** |
| HOC vs SPS | FINE (base) | 50.00 | 50.00 |
|  | Conformer A | **77.78** | **77.78** |
|  | Conformer B | 66.67 | 66.67 |
| HOC vs SAA | FINE (base) | **50.00** | **50.00** |
|  | Conformer A | **50.00** | **50.00** |
|  | Conformer B | 44.44 | 44.44 |
| HOC vs SFE | FINE (base) | **55.56** | **55.56** |
|  | Conformer A | **55.56** | **55.56** |
|  | Conformer B | 38.89 | 38.89 |
| WFE vs WAA | FINE (base) | 55.56 | 55.56 |
|  | Conformer A | 61.11 | 61.11 |
|  | Conformer B | **72.22** | **72.22** |
| WFE vs EPS | FINE (base) | 44.44 | 44.44 |
|  | Conformer A | **50.00** | **50.00** |
|  | Conformer B | 44.44 | 44.44 |
| WFE vs EFE | FINE (base) | **77.78** | **77.78** |
|  | Conformer A | 38.89 | 38.89 |
|  | Conformer B | 50.00 | 50.00 |
| WFE vs SPS | FINE (base) | 66.67 | 66.67 |
|  | Conformer A | 44.44 | 44.44 |
|  | Conformer B | **72.22** | **72.22** |
| WFE vs SAA | FINE (base) | 44.44 | 44.44 |
|  | Conformer A | 50.00 | 50.00 |
|  | Conformer B | **55.56** | **55.56** |
| WFE vs SFE | FINE (base) | 44.44 | 44.44 |
|  | Conformer A | **61.11** | **61.11** |
|  | Conformer B | 50.00 | 50.00 |
| WAA vs EPS | FINE (base) | **50.00** | **50.00** |
|  | Conformer A | **50.00** | **50.00** |
|  | Conformer B | **50.00** | **50.00** |
| WAA vs EFE | FINE (base) | **66.67** | **66.67** |
|  | Conformer A | 44.44 | 44.44 |
|  | Conformer B | 55.56 | 55.56 |
| WAA vs SPS | FINE (base) | **55.56** | **55.56** |
|  | Conformer A | 38.89 | 38.89 |
|  | Conformer B | 50.00 | 50.00 |
| WAA vs SAA | FINE (base) | **55.56** | **55.56** |
|  | Conformer A | 22.22 | 22.22 |
|  | Conformer B | 33.33 | 33.33 |
| WAA vs SFE | FINE (base) | **66.67** | **66.67** |
|  | Conformer A | 55.56 | 55.56 |
|  | Conformer B | 50.00 | 50.00 |
| EPS vs EFE | FINE (base) | 50.00 | 50.00 |
|  | Conformer A | 50.00 | 50.00 |
|  | Conformer B | **55.56** | **55.56** |
| EPS vs SPS | FINE (base) | 38.89 | 38.89 |
|  | Conformer A | **50.00** | **50.00** |
|  | Conformer B | 44.44 | 44.44 |
| EPS vs SAA | FINE (base) | 50.00 | 50.00 |
|  | Conformer A | **66.67** | **66.67** |
|  | Conformer B | 55.56 | 55.56 |
| EPS vs SFE | FINE (base) | 66.67 | 66.67 |
|  | Conformer A | 66.67 | 66.67 |
|  | Conformer B | **77.78** | **77.78** |
| EFE vs SPS | FINE (base) | 61.11 | 61.11 |
|  | Conformer A | **66.67** | **66.67** |
|  | Conformer B | 50.00 | 50.00 |
| EFE vs SAA | FINE (base) | **72.22** | **72.22** |
|  | Conformer A | 61.11 | 61.11 |
|  | Conformer B | 50.00 | 50.00 |
| EFE vs SFE | FINE (base) | 50.00 | 50.00 |
|  | Conformer A | **61.11** | **61.11** |
|  | Conformer B | 44.44 | 44.44 |
| SPS vs SAA | FINE (base) | **50.00** | **50.00** |
|  | Conformer A | 44.44 | 44.44 |
|  | Conformer B | **50.00** | **50.00** |
| SPS vs SFE | FINE (base) | **61.11** | **61.11** |
|  | Conformer A | **61.11** | **61.11** |
|  | Conformer B | 50.00 | 50.00 |
| SAA vs SFE | FINE (base) | 33.33 | 33.33 |
|  | Conformer A | 55.56 | 55.56 |
|  | Conformer B | **72.22** | **72.22** |

Subjects per cell: [1]
