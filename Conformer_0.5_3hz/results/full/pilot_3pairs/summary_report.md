# FineMI 0.5–3 Hz: FINE vs Conformer A vs Conformer B

> **Pilot snapshot (3 pairs, job 3439893).** Files referenced below are in this `pilot_3pairs/` folder. The 28-pair sweep writes its own tables to `../comparison/`.

**Runs:** SLURM job 3439893, TACC Lonestar6 `gpu-a100-dev`, 3× NVIDIA A100-PCIE-40GB, torch 2.8.0+cu128, seed 42
**Design:** within-subject, 18 subjects, 5-fold stratified CV, 50 epochs, windows 800/1500/3000/4000 ms
**Pairs:** WAA vs SAA, HOC vs WFE, EPS vs SPS (3 pairs × 3 architectures = 9 runs, all complete, 18/18 subjects in every cell)

All three models share one training loop: the same splits, z-scoring, noise augmentation, seeds, and best-validation checkpoint (deepcopy). Accuracy is the per-subject test accuracy pooled over the 5 folds (~80 test trials per subject). Stats are computed across subjects.

---

## 1. Key findings

1. **The 0.5–3 Hz band is much more informative than 8–30 Hz for these pairs.** With identical code, both conformers gained about 9–13 percentage points (pp) over their earlier 8–30 Hz runs:
   - Conformer A: 53.8% → 66.8% (+13.1 pp; 18/18 subjects improved; p < .0001)
   - Conformer B: 56.4% → 65.8% (+9.4 pp; 16/18 subjects improved; p < .0001)

   At 8–30 Hz the conformers were close to chance. At 0.5–3 Hz roughly two thirds of subject × window cells are above chance.
2. **Conformer A is the best architecture overall on 0.5–3 Hz**, averaged over windows and pairs:
   - Conformer A: 66.8%
   - Conformer B: 65.8%
   - FINE: 64.8%

   A beats FINE by +2.0 pp (p = .002; 14/18 subjects). It also has the highest window-average on every one of the 3 pairs.
3. **The ranking depends on window length.**
   - **Long windows (3000–4000 ms):** A is clearly best, +4.4 pp over FINE. At 4000 ms 17/18 subjects are better with A (p < .001).
   - **800 ms:** A is the *worst* model. B is 2.4 pp better than A (p = .005) and is the best or tied-best early decoder.
   - **1500 ms:** all three are statistically indistinguishable (66.3–67.7%).
4. **Conformer B gives only a small, fragile gain over FINE** (+1.0 pp, p = .024 with a t-test, p = .043 with Wilcoxon). No single window shows a significant pooled difference.
5. **Pair difficulty is consistent across models:** WAA/SAA is easiest (~69–71%), then EPS/SPS (~66–68%), then HOC/WFE (~60–62%, near the chance threshold).

---

## 2. Accuracy tables (0.5–3 Hz)

### 2.1 Pooled over the 3 pairs (mean of 18 subjects)

| Architecture | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| FINE (base) | 64.36 | 66.30 | 64.31 | 64.29 | 64.82 |
| Conformer A | 62.49 | 67.49 | **68.67** | **68.68** | **66.83** |
| Conformer B | **64.92** | **67.72** | 64.70 | 65.79 | 65.78 |

How accuracy changes with window length:
- **FINE:** peaks at 1500 ms, then flattens.
- **Conformer A:** rises to 3000 ms and holds.
- **Conformer B:** peaks at 1500 ms.

Only Conformer A keeps improving with more context.

### 2.2 Per pair: mean (SD across subjects)

| Pair | Architecture | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---|---:|---:|---:|---:|---:|
| WAA vs SAA | FINE (base) | **65.79 (12.10)** | 71.23 (12.72) | 68.78 (10.93) | 69.12 (10.44) | 68.73 (10.49) |
|  | Conformer A | 63.47 (8.48) | **72.65 (12.14)** | **73.60 (12.08)** | **73.15 (11.41)** | **70.72 (9.89)** |
|  | Conformer B | 65.61 (10.40) | 70.73 (12.87) | 69.44 (12.20) | 70.14 (12.55) | 68.98 (11.10) |
| HOC vs WFE | FINE (base) | 61.25 (8.45) | 60.04 (8.15) | 58.75 (9.00) | 58.60 (8.98) | 59.66 (7.21) |
|  | Conformer A | 60.81 (9.17) | 61.00 (7.71) | **62.12 (10.98)** | **62.43 (9.87)** | **61.59 (8.48)** |
|  | Conformer B | **63.22 (5.98)** | **63.50 (7.70)** | 57.46 (9.72) | 59.84 (9.26) | 61.00 (7.20) |
| EPS vs SPS | FINE (base) | **66.05 (11.93)** | 67.62 (12.63) | 65.40 (10.43) | 65.15 (9.85) | 66.06 (9.65) |
|  | Conformer A | 63.18 (12.16) | 68.83 (13.16) | **70.29 (11.12)** | **70.46 (10.74)** | **68.19 (10.68)** |
|  | Conformer B | 65.93 (10.37) | **68.93 (11.30)** | 67.19 (11.27) | 67.39 (10.22) | 67.36 (9.69) |

The best single result is Conformer A on WAA vs SAA at 3000 ms: 73.6%.

---

## 3. Paired comparisons between architectures (0.5–3 Hz)

Each cell shows:
- the difference in percentage points,
- the paired t-test p-value (df = 17),
- the number of subjects where the first model was better vs worse ("wins–losses").

**"Pooled" means each subject's accuracy is averaged over the 3 pairs first.** Wilcoxon signed-rank p-values, SDs and effect sizes are in `paired_stats.csv` (in this folder). The Wilcoxon results agree with the t-test on every result highlighted below.

### 3.1 Pooled over pairs, by window

| Contrast | 800 ms | 1500 ms | 3000 ms | 4000 ms | Avg |
|---|---:|---:|---:|---:|---:|
| A − FINE | −1.88 (p=.19, 8–10) | +1.20 (p=.17, 13–5) | **+4.36 (p=.001, 14–4)** | **+4.39 (p<.001, 17–1)** | **+2.02 (p=.002, 14–4)** |
| B − FINE | +0.56 (p=.62, 10–7) | +1.42 (p=.058, 11–5) | +0.39 (p=.64, 9–8) | +1.50 (p=.059, 13–4) | +0.97 (p=.024, 11–7) |
| A − B | **−2.43 (p=.005, 4–14)** | −0.23 (p=.74, 9–7) | **+3.97 (p=.001, 13–4)** | +2.89 (p=.008, 13–5) | +1.05 (p=.098, 13–5) |

**Multiple comparisons (Holm correction).** Applied over the 12 window-level tests above, four results survive:
- A > FINE at 3000 ms
- A > FINE at 4000 ms
- A > B at 3000 ms
- B > A at 800 ms

Applied over the 3 window-average tests:
- **A − FINE** survives.
- **B − FINE** survives only with the t-test (p = .024 against a threshold of .025). It does not survive with Wilcoxon (p = .043).
- **A − B** does not survive.

### 3.2 Per pair, window-average

| Pair | A − FINE | B − FINE | A − B |
|---|---:|---:|---:|
| WAA vs SAA | +1.98 (p=.057, 12–6) | +0.25 (p=.76, 11–7) | +1.74 (p=.097, 12–6) |
| HOC vs WFE | +1.93 (p=.032, 14–4) | +1.34 (p=.039, 13–4) | +0.59 (p=.36, 9–7) |
| EPS vs SPS | +2.14 (p=.049, 14–4) | +1.31 (p=.046, 11–5) | +0.83 (p=.48, 10–8) |

The direction is the same in every pair (A > B > FINE). With only 18 subjects per pair, the pair-level tests are borderline individually; the pooled test is what gives the result its strength.

---

## 4. 0.5–3 Hz vs 8–30 Hz (conformers only)

The 8–30 Hz numbers come from `ConformerEEG/experiments/option_{a,b}/results/full`: the same model code, hyperparameters, CV, seeds, deepcopy checkpoint and A100 type. The only change is the input data (`FineMI` 8–30 Hz vs `FineMI_0.5_3hz`).

Window-average accuracy (%), with subjects better at 0.5–3 Hz:

| Pair | Conformer A 8–30 → 0.5–3 | Conformer B 8–30 → 0.5–3 |
|---|---:|---:|
| WAA vs SAA | 55.97 → 70.72 (**+14.7**, 17/18) | 59.19 → 68.98 (**+9.8**, 13/18) |
| HOC vs WFE | 52.13 → 61.59 (**+9.5**, 15/18) | 53.28 → 61.00 (**+7.7**, 15/18) |
| EPS vs SPS | 53.21 → 68.19 (**+15.0**, 17/18) | 56.67 → 67.36 (**+10.7**, 16/18) |
| **Pooled** | 53.77 → 66.83 (**+13.1**, 18/18, p<.0001, dz=1.77) | 56.38 → 65.78 (**+9.4**, 16/18, p<.0001, dz=1.35) |

Every pair × window band contrast is positive, and all are p < .05 by t-test. The weakest is Conformer B, HOC vs WFE, 3000 ms (t-test p = .043; Wilcoxon p = .052, just above .05).

**The ranking of the two conformers flips between bands.** At 8–30 Hz Conformer B was better than A (56.4 vs 53.8). At 0.5–3 Hz Conformer A is better. A pure temporal-conv + transformer model seems to benefit most from the slow, low-frequency activity in this band.

### Subjects above chance

A subject counts as above chance at ≥ 60% accuracy, which is roughly p < .05 on a one-sided binomial test with 80–90 test trials. There are 216 subject × pair × window cells per model; by chance alone about 11 would pass.

| | FINE | Conformer A | Conformer B |
|---|---:|---:|---:|
| 0.5–3 Hz | 139 (64%) | 149 (69%) | **153 (71%)** |
| 8–30 Hz | not re-run | 43 (20%) | 67 (31%) |

At 0.5–3 Hz, Conformer B has the most above-chance cells despite a lower mean than A, because of its stronger short windows. HOC vs WFE is the weakest pair: only 7–13 of 18 subjects are above chance for any model and window.

---

## 5. Caveats

- **Only 3 of 28 pairs.** The architecture ranking and the size of the band effect may not hold for the other 25 pairs.
- **One seed.** Differences of 1–2 pp (B vs FINE, and A vs B at the window average) are within what a different seed could plausibly change. The large effects (the band effect, and A at long windows) are unlikely to be seed artifacts.
- **Subjects are not independent across pairs.** The pooled tests average each subject over the 3 pairs first, so n = 18 throughout; pairs were never treated as extra samples.
- **FINE numbers are not the notebook's numbers.** FINE here tests the best-validation checkpoint (deepcopy). The EMBC notebook's shallow `state_dict().copy()` tested last-epoch weights. For example, the notebook's 8–30 Hz WAA/SAA FINE means were 63.0 / 61.4 / 59.5 / 61.0; those are not directly comparable to the 65.8 / 71.2 / 68.8 / 69.1 here. To get a clean FINE band comparison, FINE would need to be re-run on 8–30 Hz with this code.
- **The band comparison assumes both datasets were preprocessed identically apart from the band-pass filter.** The epoch shape is the same (62 channels × 1126 samples at 250 Hz).
- **Runtime not recorded here.** The worker logs (`results/full/logs/*.log`) are excluded by the repo's `.gitignore` (`*.log`), so they were not pushed.

## 6. Suggested next steps

1. **Run all 28 pairs on 0.5–3 Hz** with Conformer A and FINE first. This job fit 3 pairs × 3 models inside the 2 h dev limit, so all 28 pairs would take 10 dev jobs (`PAIRS` in groups of 3), or one longer `gpu-a100` job.
2. **Re-run FINE on 8–30 Hz with this pipeline**, to confirm the band effect holds for the baseline too.
3. **Try more seeds** (e.g. 3) on these 3 pairs, to check the small differences (B vs FINE, A vs B).
4. **Early decoding (≤ 1500 ms):** Conformer B or FINE are the better choices. Conformer A's short-window weakness may come from its pooling: a 75-sample kernel with stride 15 leaves only 9 tokens at 800 ms, so a smaller pool is worth trying for short windows.

---

**Files:**
- `comparison_table.md` and `comparison_{mean,sd,long}.csv`: accuracy tables
- `paired_stats.csv` (in this folder): all 100 paired tests (t-test, Wilcoxon, Cohen's dz, win counts)
- `all_subjects_long.csv`: per-subject accuracies
- `../<model>/<PAIR>/`: per-subject and per-fold results
