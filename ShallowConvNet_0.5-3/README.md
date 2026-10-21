# ShallowConvNet_0.5-3

ShallowConvNet (Schirrmeister et al. 2017) on the fixation-cross-baseline, imagery-only
FineMI data, **0.5-3 Hz**, at the full 250 Hz (no decimation). The sister folder
`ShallowConvNet_8-30` is identical apart from the band block at the top of
`src/config.py`.

| | |
|---|---|
| Data | `fixationcrossbaseline_0.5_3hz_MIonly`: 0 to 4 s after "Start Imagining", 1000 samples, baseline -4 to -2 s |
| Windows | first 800 / 1500 / 3000 / 4000 ms after imagery onset = 200 / 375 / 750 / 1000 samples |
| Model | temporal conv 40 x 25 (100 ms), spatial conv 40 x 62, BN, square, mean pool 75 / stride 15, log, dropout 0.5, conv classifier |
| Parameters | 100,882 / 101,842 / 103,842 / 105,202 at the four windows (the classifier grows with the window) |
| Training | the same loop as LEAN / EEG Conformer in `Fine Conformer Fixation Cross Fix`: 5-fold CV per subject with the same seeded splits, z-score, noise augmentation, Adam 1e-3, 50 epochs, best-validation checkpoint |

The training is the same as LEAN's and EEG Conformer's, not Schirrmeister's
original recipe (cropped training, AdamW, cosine schedule). That keeps the results
paired fold by fold with theirs. `summarize.py` then compares ShallowConvNet with both
of them, paired over subjects, using the full-mode results committed in
`Fine Conformer Fixation Cross Fix/results/full/`.

## Layout

```
run_vast.sh                    the whole run on one Vast.ai instance (smoke | full)
src/config.py                  band, data path, windows, model and training settings
src/models/shallowconvnet.py   the model
src/train.py                   training loop
scripts/preflight.py           packages, GPU, data (1000-sample epochs, power in band),
                               parameter counts, determinism, time projection
scripts/run_selected_pairs.py  trains pairs; finished pairs are skipped
scripts/check_results.py       every expected file, subject, fold and parameter count
scripts/summarize.py           ShallowConvNet's numbers, and its paired differences
                               from the reference runs
```

## Vast.ai

1. **On your PC**, push: `git add ShallowConvNet_0.5-3 && git commit -m "ShallowConvNet 0.5-3 Hz" && git push`
2. **Rent** a PyTorch/CUDA 12.x instance. 1-4 GPUs (any 24 GB+ card), at least 30 GB
   disk, and about **4 GB of RAM per GPU**: every worker holds all 18 subjects at 250 Hz.
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
   tmux new -s shallow03                 # detach Ctrl-b d, reattach: tmux attach -t shallow03
   cd /workspace/FineMI-enhancements/ShallowConvNet_0.5-3
   bash run_vast.sh smoke                # must end with SMOKE TEST PASSED
   bash run_vast.sh full                 # rerun the same command to resume
   ```
   In the smoke preflight, check that `data` reports `T=1000 @ 250 Hz` (warnings only for
   subjects 1 and 5) and `band` reports most of the power in 0.5-3 Hz. Also note the
   `timing projection`.
5. **Download** from your PC, then destroy the instance:
   ```bash
   cd "/c/Users/rmaah/Documents/Summer 2026/Fine MI/FineMI-enhancements/ShallowConvNet_0.5-3"
   scp -P <PORT> -r root@<IP>:/workspace/FineMI-enhancements/ShallowConvNet_0.5-3/results .
   ```
   The results are in `results/full/summary/summary.md`.
