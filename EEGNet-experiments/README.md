# EEGNet-experiments

EEGNet-8,2 (Lawhern et al. 2018) on the fixation-cross-baseline, imagery-only FineMI
data, **0.5-3 Hz**, at two sampling rates trained side by side:

| Model | Rate | Temporal kernel | Pool 1 / sep. kernel / pool 2 | Parameters (800 / 1500 / 3000 / 4000 ms) |
|---|---|---|---|---|
| `eegnet_250` | 250 Hz, no decimation (as ShallowConvNet) | 125 (0.5 s) | 4 / 16 / 8 | 2,778 / 2,938 / 3,322 / 3,578 |
| `eegnet_50` | 50 Hz, D = 5 (as LEAN and EEG Conformer) | 25 (0.5 s) | 2 / 8 / 4 | 1,818 / 1,946 / 2,234 / 2,458 |

| | |
|---|---|
| Data | `fixationcrossbaseline_0.5_3hz_MIonly`: 0 to 4 s after "Start Imagining", 1000 samples at 250 Hz, baseline -4 to -2 s |
| Windows | first 800 / 1500 / 3000 / 4000 ms after imagery onset = 200 / 375 / 750 / 1000 samples at 250 Hz, 40 / 75 / 150 / 200 at 50 Hz |
| Model | F1 8, D 2, F2 16; temporal conv, depthwise spatial conv (max-norm 1.0), BN, ELU, avg pool, dropout 0.5; separable conv, BN, ELU, avg pool, dropout 0.5; linear classifier (max-norm 0.25). Outputs logits (no double softmax). BN eps 1e-3 as in Keras, momentum 0.1 as in the ShallowConvNet runs |
| Training | the same loop as LEAN / EEG Conformer / ShallowConvNet: 5-fold CV per subject with the same seeded splits, z-score, noise augmentation, Adam 1e-3, 50 epochs, best-validation checkpoint |

The FineMI authors' code has no EEGNet, so the model follows the original paper. The
temporal kernel is half the sampling rate at both rates, as the paper prescribes.

`summarize.py` compares, paired over subjects:
- EEGNet 250 Hz against EEGNet 50 Hz
- each of them against LEAN and EEG Conformer (`Fine Conformer Fixation Cross Fix/results/full/`)
  and ShallowConvNet (`ShallowConvNet_0.5-3/results/full/`)

## Layout

```
run_vast.sh                    the whole run on one Vast.ai instance (smoke | full)
src/config.py                  data path, windows, per-model rate and EEGNet settings,
                               expected parameter counts
src/models/eegnet.py           the model
src/train.py                   training loop
scripts/preflight.py           packages, GPU, data at both rates (1000 / 200-sample
                               epochs, power in band), parameter counts, determinism,
                               time projection
scripts/run_selected_pairs.py  trains pairs; finished pairs are skipped
scripts/check_results.py       every expected file, subject, fold and parameter count
scripts/summarize.py           both models' numbers, and the paired comparisons
```

## Vast.ai

1. **On your PC**, push: `git add EEGNet-experiments && git commit -m "EEGNet 0.5-3 Hz at 250 and 50 Hz" && git push`
2. **Rent** a PyTorch/CUDA 12.x instance. 1-4 GPUs (any 16 GB+ card is plenty;
   EEGNet is tiny), at least 30 GB disk, and about **5 GB of RAM per GPU**: every
   worker holds all 18 subjects at 250 Hz and at 50 Hz.
3. **Upload the data** from your PC:
   ```bash
   cd "/c/Users/rmaah/Documents/Summer 2026/Fine MI"
   scp -P <PORT> -r fixationcrossbaseline_0.5_3hz_MIonly root@<IP>:/workspace/
   ```
4. **On the instance**:
   ```bash
   cd /workspace
   git clone https://github.com/rmaahin/FineMI-enhancements.git     # or: git pull
   python -m venv /workspace/venv --system-site-packages
   source /workspace/venv/bin/activate
   pip install numpy==2.0.2 scipy==1.13.1 scikit-learn==1.6.1 pandas==2.3.3
   tmux new -s eegnet                    # detach Ctrl-b d, reattach: tmux attach -t eegnet
   cd /workspace/FineMI-enhancements/EEGNet-experiments
   bash run_vast.sh smoke                # must end with SMOKE TEST PASSED
   bash run_vast.sh full                 # rerun the same command to resume
   ```
   In the smoke preflight, check that `data` reports `T=1000 @ 250 Hz` and
   `T=200 @ 50 Hz` (warnings only for subjects 1 and 5), that `band` reports most of the
   power in 0.5-3 Hz, and that both models pass the parameter check. Also note the
   `timing projection`. To run one rate only: `MODELS=eegnet_50 bash run_vast.sh full`.
5. **Download** from your PC, then destroy the instance:
   ```bash
   cd "/c/Users/rmaah/Documents/Summer 2026/Fine MI/FineMI-enhancements/EEGNet-experiments"
   scp -P <PORT> -r root@<IP>:/workspace/FineMI-enhancements/EEGNet-experiments/results .
   ```
   The results are in `results/full/summary/summary.md`.
