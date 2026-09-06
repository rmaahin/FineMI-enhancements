# FBCNet — Technique 1: filter-bank spectral filtering

Adds the FBCNet-style filter-bank front-end from *Proposed Feature Extraction
Techniques* to the FINE architecture in `EMBC_deterministic-3.ipynb`. Nothing in the
original pipeline was changed; only the front-end is new.

Layout mirrors `../morlet-scalogram/` (Technique 2) file-for-file, so the two techniques
can be run and aggregated the same way.

```
fine_mi.py     pipeline + filter bank + models     (import this BEFORE torch)
run_local.py   batch driver over the 28 task pairs
```

## Front-end

Six fixed 4 Hz sub-bands over 8–30 Hz (`8-12, 12-16, 16-20, 20-24, 24-28, 28-32`),
each a linear-phase Hamming-windowed-sinc bandpass FIR (413 taps @ 250 Hz, ~2 Hz
transition, ≥53 dB stopband, unit gain at band centre). Applied by FFT convolution on
the GPU, group delay removed, then per-(channel, band) z-scored with training-fold
statistics.

```
(N, C, T)  ->  (N, C, B, T)      B = 6,  T = 1000  (4000 ms @ 250 Hz)
```

The band axis sits at dim 2 rather than the proposal's `N × B × C × T` so that `C`
stays the conv in-channel axis and `(B, T)` is the 2D feature map — the same convention
the Morlet arm uses for `(F, T)`. Same tensor, transposed.

Zero trainable parameters: the bank is a fixed transform applied on the data side,
before the `DataLoader`.

## Model

`FINEFilterBank2D` is the original FINE graph with `Conv1d → Conv2d` and the band axis
added:

| stage | original (1D) | filter-bank arm (2D) |
|---|---|---|
| MTC | `Conv1d` k = 7/15/31 | `Conv2d` k = (1, 7/15/31) — **per band**, no mixing |
| spatial | `Conv1d` k = 3 | `Conv2d` k = 3×3 — where bands mix (per-band spatial filters) |
| pool | `MaxPool1d(2)` | `MaxPool2d(2)` |
| fusion | depthwise-separable + ECA | same, 2D |
| global pool | `AdaptiveAvgPool1d(1)` | `mean(-2, -1)` — deterministic |
| head | Linear(128)+ReLU+Drop(.4)+Linear | unchanged |

`AdaptiveAvgPool2d`/`AdaptiveMaxPool2d` have nondeterministic CUDA backwards and *raise*
under `use_deterministic_algorithms(True)`, which this pipeline sets — hence
`mean`/`amax`.

Because the filter bank does **not** decimate time, the MTC kernels cover exactly the
same 28/60/124 ms spans as in the raw arm. The front-end is the only thing that changes
between arms (the CWT arm could not say this — it decimates 4×).

## Running

```bash
python run_local.py --selftest                    # front-end + model checks, no training
python run_local.py --mode fbc --pairs HOC/SPS --smoke
python run_local.py --mode fbc --pairs all        # filter-bank arm, 28 pairs
python run_local.py --mode raw --pairs all        # verbatim baseline, same folds
python run_local.py --mode both --pairs all --time-probe
```

`--mode raw` is bit-for-bit the original notebook's method and shares fold indices,
seeds, normalisation and augmentation draws with `fbc`, so the two are paired per
subject and per fold.

Results land in `mi_results_fbc/<mode>/<PAIR>/` as `per_subject.csv`, `per_fold.csv`,
`wide_subject_x_window.csv`, `predictions.csv`, `results.json`, `results.pkl` — same
format as the other arms. Already-saved pairs are skipped unless the stamped
`pipeline_version`, cue alignment, band layout, FIR length or normalisation differ, so
a sweep can be interrupted and resumed freely.

## Diagnostic knobs (env vars; defaults reproduce the setup above)

| var | default | effect |
|---|---|---|
| `FINE_FB_BANDS` | `8-12,...,28-32` | band layout, e.g. `8-13,13-20,20-30` |
| `FINE_FB_TRANS` | `2.0` | FIR transition width in Hz → filter length |
| `FINE_FB_NORM` | `bandz` | `none` keeps raw sub-band amplitudes |
| `FINE_FB_CHUNK` | auto | filter batch size; lower if VRAM is tight |
| `FINE_MTC_KERNELS_2D` | `7,15,31` | MTC temporal kernels |
| `FINE_CUE_SAMPLE` | `0` | `125` re-runs cue-aligned (original slices from 0) |

Point diagnostics at their own `--results-root` so they never mix with the main sweep.

## Verified

`make_bandpass_fir` and the FFT-convolution alignment were checked offline in NumPy:
taps exactly symmetric, DC/Nyquist gain ≤ 1.6e-3, stopband ≤ −53 dB, unit centre gain,
10/18/26 Hz sines landing in the correct band ~30× above the runner-up, group delay
removed to 1e-14, and the six sub-bands summing back to a band-limited input with a
5.5 % residual (the transition-band ripple). The remaining checks — torch shapes,
forward/backward under the determinism flags, chunk invariance, VRAM — run in
`--selftest`; this machine has no `torch`, so run it once on the GPU box before the
sweep.
