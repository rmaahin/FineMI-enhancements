# LEAN_v2

LEAN (ICASSP 2027 paper, 25,752 parameters) rebuilt step by step with ideas from
EEGNet, EEG-TCNet, LMDA-Net and MBCL, to beat EEGNet-8,2 (63.65% / 63.41% window
average at 250 / 50 Hz) on the fixation-cross-baseline, imagery-only **0.5-3 Hz** FineMI
data at **50 Hz**. Same data, CV folds, seeds, z-score and noise augmentation as LEAN,
EEG Conformer and EEGNet, so every result pairs with theirs subject by subject.

## Presets

One configurable network (`src/models/lean_v2.py`) covers every variant; a preset =
an `arch` dict + a `train` dict in `src/config.py`.

**Ladder (cumulative: each rung = previous rung + one change)**

| Preset | Adds | Params |
|---|---|---|
| `lean_v2_r0` | LEAN exactly (same layer order -> same initial weights) | 25,752 |
| `lean_v2_r1` | separable fusion (EEGNet block 2) with **average** pools 2/4 instead of conv k3 + max pool 3 -> 160 ms tokens | 21,752 |
| `lean_v2_r2` | transformer -> **TCN** (3 blocks, k 4, dilations 1/2/4, 43-token receptive field) and GAP + MLP head -> **last-step linear readout**; width 32 -> 16 | 14,440 |
| `lean_v2_r3` | **shared temporal filters** (4 per branch, 7/15/31) + depthwise spatial conv (D = 2) instead of per-electrode filters + 1x1 mix | 9,278 |
| `lean_v2_r4` | **LMDA depth attention** between the temporal and spatial convs | 9,286 |
| `lean_v2_r5` | ELU, max-norm (spatial 1.0, readout 0.25), dropout 0.25 after pools; **fixed schedule**: cosine LR, average of the last 10 epochs' weights, no best-validation checkpoint | 9,286 |
| `lean_v2_r6` | + supervised contrastive loss (weight 0.5, tau 0.05, training-only 16-d projection) | 9,286 |
| `lean_v2_r7` | + crops: train on random 90% crops, test on the mean of 3 crops (**final model**) | 9,286 |

The feature-refinement block (DS conv k5 + channel gate, paper Sec. 2.5) is kept in
every rung. All parameter counts are window-independent except where noted below.

**Leave-one-out** (`loo`): R7 with one change removed - `lean_v2_loo_fusion` (9,862),
`_tcn` (transformer, 6,614), `_readout` (GAP + MLP, 10,918), `_shared` (per-electrode
front, also drops depth attention, 14,440), `_depthatt` (9,278), `_reg` (9,286),
`_schedule` (best-val, 9,286), `_supcon` (9,286). R7 minus crops is R6.

**Training only** (`training`), on LEAN's architecture: `lean_schedule`, `lean_supcon`
(25,752), `lean_recipe` (all of R7's training changes; 25,560, since crops shorten the
positional embedding).

**Baselines** (`baselines`): `eegnet_50` (EEGNet-experiments' eegnet_50, rerun here so
it shares the GPU type) and `eegnet_50_recipe` (with R7's training recipe;
1,786 / 1,914 / 2,170 / 2,362 parameters, crops shorten the classifier).

**R5 study** (`r5`, added after the ladder screen: R5 65.17% vs EEGNet 63.60%, SupCon and
crops hurt): `lean_v2_r4_sched` (R4 + fixed schedule only) and `lean_v2_r4_reg` (R4 +
ELU/max-norm/dropout only) split R5's two changes; `lean_v2_r5_loo_{fusion, tcn,
readout, shared, depthatt}` remove one architecture change from R5 (9,862 / 6,662 /
10,918 / 14,440 / 9,278); `lean_r5recipe` (LEAN + R5's regularisation and schedule,
25,752) and `eegnet_50_sched` (EEGNet + fixed schedule) give the references the same
recipe. The group also lists `lean_v2_r0/r4/r5`, `lean_schedule` and `eegnet_50`, so
the comparisons run on the same GPU; finished runs are skipped.

**Final** (`final`, chosen from the R5 screen): `lean_v2_final` = R5 without depth
attention (9,278 parameters), and `eegnet_50_sched` (EEGNet with the same fixed
schedule); both are also compared with the EEGNet 50 Hz and LEAN paper runs.

## Modes

| Mode | Subjects | Folds | Epochs | Windows | Pairs |
|---|---|---|---|---|---|
| smoke | 2 | 2 | 2 | 800/1500/3000/4000 | 3 |
| screen | 18 | 5 | preset (50) | 800, 4000 | 6: HOC/SPS, WAA/SFE (distant), HOC/WFE, EPS/SAA (adjacent), EPS/EFE, SPS/SAA (same-joint) |
| full | 18 | 5 | preset (50) | 800/1500/3000/4000 | 28 |

`results/<mode>/summary/summary.md` has the mean accuracies, the ladder steps (R1-R0,
R2-R1, ...), leave-one-out, training-only and baseline comparisons, every preset minus
EEGNet (paper run), by joint group and per pair. All comparisons are paired over
subjects with Holm-corrected p. Screen and full compare against the **full** LEAN and
EEGNet results in `../Fine Conformer Fixation Cross Fix` and `../EEGNet-experiments`,
restricted to the same pairs and windows.

## Run on Vast.ai (2x RTX 3090)

1. **Push from your PC**: `git add LEAN_v2 && git commit -m "LEAN_v2" && git push`
2. **Rent** a PyTorch / CUDA 12.x instance with 2 GPUs, 30 GB disk, ~16 GB RAM
   (about 1.5 GB per worker).
3. **Upload the data** from your PC:
   ```bash
   cd "/c/Users/rmaah/Documents/Summer 2026/Fine MI"
   scp -P <PORT> -r fixationcrossbaseline_0.5_3hz_MIonly root@<IP>:/workspace/
   ```
4. **On the instance**:
   ```bash
   cd /workspace
   git clone https://github.com/rmaahin/FineMI-enhancements.git      # or: git pull
   python -m venv /workspace/venv --system-site-packages
   source /workspace/venv/bin/activate
   pip install numpy==2.0.2 scipy==1.13.1 scikit-learn==1.6.1 pandas==2.3.3
   tmux new -s lean                       # detach Ctrl-b d, reattach: tmux attach -t lean
   cd /workspace/FineMI-enhancements/LEAN_v2

   WORKERS_PER_GPU=3 bash run_vast.sh smoke     # must end with SMOKE TEST PASSED
   WORKERS_PER_GPU=3 bash run_vast.sh screen    # the ladder (R0-R7)
   MODELS=r5 WORKERS_PER_GPU=3 bash run_vast.sh screen     # decompose R5
   MODELS="loo training baselines" WORKERS_PER_GPU=3 bash run_vast.sh screen
   MODELS=final WORKERS_PER_GPU=3 bash run_vast.sh full    # lean_v2_final + eegnet_50_sched, 28 pairs x 4 windows
   ```
   In the smoke preflight check that `LEAN layout (lean_v2_r0)` passes, that every
   `model ...` line shows the predicted parameters, and read the `timing projection`
   (screen and full hours for the presets you ran). Pick the full-run presets from
   the screen summary. To stop a run: Ctrl-C AND `pkill -f scripts.run_selected_pairs`.
5. **Download** from your PC, then destroy the instance:
   ```bash
   cd "/c/Users/rmaah/Documents/Summer 2026/Fine MI/FineMI-enhancements/LEAN_v2"
   scp -P <PORT> -r root@<IP>:/workspace/FineMI-enhancements/LEAN_v2/results .
   ```

## Notes

- **Determinism** holds per GPU model; R0 reproduces the paper's LEAN numbers exactly
  only on the GPU type LEAN ran on (A100). On the 3090s, compare the ladder against
  its own R0, and read "R0 vs LEAN (paper run)" as a hardware check.
- **Fixed schedule** keeps the 25% validation split out of training, so every preset
  trains on exactly the same trials; the split is only used for logging.
- **Not yet built**: R8 (cross-subject pre-training, then per-subject fine-tuning).
  It needs its own pre-training stage and comes after the screen results.

## Layout

```
run_vast.sh                    smoke | screen | full on one Vast.ai instance
src/config.py                  presets (arch + train), groups, modes, references
src/models/lean_v2.py          the configurable network; blocks.py its layers
src/models/eegnet.py           EEGNet-8,2 (with a features output for SupCon)
src/param_budget.py            hand-derived parameter counts (asserted everywhere)
src/train.py                   training loop: best-val / fixed-average schedules,
                               SupCon, crops
scripts/preflight.py           data, LEAN layout, every preset's parameters and
                               training path, determinism, time projection
scripts/run_selected_pairs.py  trains (pair, preset) runs; finished ones are skipped
scripts/check_results.py       every file, subject, fold and parameter count
scripts/summarize.py           tables and paired comparisons
```
