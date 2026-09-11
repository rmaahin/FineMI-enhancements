# Riemannian — Technique 3: SPD covariance features

Adds per-trial spatial covariance matrices, handled with the affine-invariant Riemannian
geometry of the SPD manifold, to the FINE architecture in `EMBC_deterministic-3.ipynb`.
The training recipe (folds, seeds, z-score, augmentation, optimiser, epochs, checkpoint
quirk, save format) is the original, verbatim. Layout mirrors `../FBCNet/`.

```
fine_mi.py          pipeline + SPD geometry + front-end + models  (import BEFORE torch)
run_local.py        batch driver over the 28 task pairs; --selftest
analyze_results.py  every arm vs raw, paired; pulls in the fbc/cwt arms if present
run_riemannian.sh   the whole pipeline: preflight -> selftest -> probe -> sweep -> analysis
submit_tacc.slurm   sbatch wrapper for a TACC A100 node
```

## Running

```bash
./run_riemannian.sh               # full: reuse raw, 6 arms x 2 channel sets x 28 pairs, analysis
CHANNEL_SETS=all ./run_riemannian.sh      # only the 62-channel versions
./run_riemannian.sh smoke         # 2 pairs x all arms x 2 subjects x 5 epochs, unsaved
./run_riemannian.sh probe         # one real pair per arm + projected sweep time
./run_riemannian.sh analyze       # rebuild tables from ./results
ARMS="ts tsimg" ./run_riemannian.sh      # subset of arms
sbatch submit_tacc.slurm                  # on TACC
```

Interrupt-safe: finished pairs are skipped unless their stamped config differs.
Everything lands in `./results/<arm>/<PAIR>/` (same files as the other techniques) and
`./results/analysis/`.

**Raw baseline is reused.** The `raw` arm is the FBCNet/Morlet raw arm bit-for-bit
(verified: identical input tensors for every fold), so `run_riemannian.sh` copies
`../FBCNet/results/raw` instead of retraining it (`REUSE_RAW=0` to recompute). Those
were produced on an A100 — run this sweep on the same GPU model and the pairing is
exact; `analyze_results.py` reports whether it is.

## Arms

| arm | features | classifier | maps to |
|---|---|---|---|
| `raw` | 8–30 Hz EEG `(62, 1000)` | FINE 1D, unchanged | baseline |
| `ts` | tangent vector `(1891,)` | FINE head (projection + 2-layer classifier) | proposal fit (a) |
| `tsimg` | tangent matrix as image `(62, 62)` | FINE 1D trunk, **unchanged** | proposal fit (b), `N×C×C` |
| `fbts` | 3 band blocks, `(5673,)` | FINE head | block version of (a) |
| `fbtsimg` | 3 band blocks, `(62, 3, 62)` | FINE 2D trunk (as in FBCNet arm) | block version of (b), Tensor-CSPNet-style |
| `mdrm` | covariance | minimum distance to Riemannian mean | "use Riemannian distances directly" (Barachant 2012) |
| `tslda` | tangent vector | shrinkage LDA | Barachant 2012 TSLDA |

`mdrm`/`tslda` have no network, run on CPU in ~1.5 min/pair, and use the same folds —
they show what the geometry does on its own, without FINE.

**Two channel sets, both pre-specified.** Every Riemannian arm runs on all 62 channels
(`results/`) and on Tensor-CSPNet's 20 sensorimotor channels (`results_motor/`,
reported as `ts@motor` etc.). Both are motivated by the papers (below), not picked from
our results; the analysis Holm-corrects across all arms, so running both is not a free
second chance. First signal, HOC/SPS only (18 subjects, laptop RTX 5070):

| arm | 62 ch | 20 motor ch |
|---|---|---|
| `ts` | 62.19 | **67.82** |
| `tsimg` | 58.90 | – |
| `fbts` | 57.57 | – |
| `fbtsimg` | 58.14 | – |
| `mdrm` | 56.95 | 61.59 |
| `tslda` | – | 65.02 |
| raw FINE (A100) | 64.23 | |

One pair is not evidence; the 28-pair sweep is.

For `tsimg` the `(C, C)` image goes into the original `CNNEarlyClassificationModel`
untouched: rows are the Conv1d in-channels exactly as electrodes are for raw EEG, and
the 7/15/31 MTC kernels slide along the partner-electrode axis instead of time.

## The geometry, per fold (all fitted on the clean training fold only)

1. **Rank.** FineMI is common-average referenced, so every 62×62 covariance is
   singular (rank 61 — smallest pooled eigenvalue is 1e-17 of the largest). `logm` and
   `M^-1/2` are undefined there, so all geometry runs on `Uᵀx`, where `U` (62×61) is the
   signal subspace of the clean training trials. The noise augmentation is per-channel
   and breaks the CAR constraint, which is why `U` must come from the clean trials.
2. **Covariance** with Ledoit-Wolf shrinkage (Congedo 2017; needed — a 5 Hz band over
   4 s has ~40 degrees of freedom per channel against P = 61).
3. **Reference** = Karcher mean of the training covariances (Barachant 2012 fixed point,
   log-Euclidean init). Val/test are mapped at the training reference — no leakage;
   within-subject CV makes this per-subject re-centering too.
4. **Tangent map** `S = logm(M^-1/2 C M^-1/2)`; vectorised as the upper triangle with
   off-diagonals × √2 so `‖vec S‖ = δ_R(C, M)`. The image is `U S Uᵀ` back in electrode
   coordinates (basis-independent, norm-preserving).
5. Every feature/pixel z-scored with training-fold statistics, like the original.

Geometry is float64 NumPy on the CPU with one BLAS thread: bit-reproducible and
independent of the GPU (and ~7× faster than multithreaded BLAS for 61×61 eigh).

## What the papers say to expect

- **Congedo et al. 2017, §4:** with N ≥ 32 electrodes the Riemannian distance is
  dominated by task-irrelevant components, and CSP-style reduction wins over plain
  MDM. 62 channels is squarely in that regime, so `mdrm` near chance on all 62 channels
  is expected, not a bug. The tangent-space arms let a learned model down-weight those
  components; `FINE_RIEM_CHANNELS=motor` reduces N instead.
- **Ju & Guan 2022 (Tensor-CSPNet):** on the 62-channel KU dataset they kept 20
  motor-cortex channels (the `motor` set here) — and there, too, pyRiemann's MDM/TSM
  were ≈50–55 %. The gain came from the frequency segmentation plus learned spatial
  reduction; their own ablation found no significant average benefit from temporal
  segmentation in CV (hence `FINE_RIEM_SEGMENTS=1` by default).
- **Barachant 2012:** TSLDA > MDRM; tangent space + a regularised linear model is the
  strong classical baseline.

## Diagnostic knobs (env vars; give each its own `RESULTS_DIR`)

| var | default | effect |
|---|---|---|
| `FINE_RIEM_CHANNELS` | `all` | `motor` = Tensor-CSPNet's 20 sensorimotor channels, or a comma list |
| `FINE_RIEM_BANDS` | `8-13,13-20,20-30` | fb* sub-bands; `8-12,12-16,16-20,20-24,24-28` = Tensor-CSPNet style |
| `FINE_RIEM_SEGMENTS` | `1` | fb* temporal segments per band (Tensor-CSPNet tensor stacking) |
| `FINE_RIEM_SHRINK` | `lw` | Ledoit-Wolf, or a fixed intensity in [0, 1) |
| `FINE_FB_TRANS` | `2.0` | FIR transition width (Hz) → filter length |
| `FINE_CUE_SAMPLE` | `0` | `125` = cue-aligned window (original slices from epoch start) |

Channel sets are driven by `CHANNEL_SETS` in the bash script (each set gets its own
`results_<set>/`). For the other knobs use a separate `RESULTS_DIR`:

```bash
FINE_RIEM_BANDS="8-12,12-16,16-20,20-24,24-28" ARMS="fbts fbtsimg" \
    RESULTS_DIR=$PWD/results_tcsp ./run_riemannian.sh
```

## Verified

`run_local.py --selftest` (run by the bash script before every sweep) checks: exp/log/sqrt
round trips; affine invariance of δ_R (1e-14); Karcher mean exact on commuting matrices,
zero gradient at the mean, order-invariant; tangent-vector norm = Riemannian distance;
Ledoit-Wolf matches sklearn to 1e-10; rank 61 detected on real data and all block
covariances SPD; filter bank band selectivity and zero phase; augmented trials stay on
the clean manifold region; every arm bit-reproducible, finite, forward/backward under
the determinism flags. The `raw` arm's input tensors were checked bit-identical to the
FBCNet raw arm for every fold.
