# Fine Conformer Fixation Cross Fix

Reruns the two ICASSP 2027 models on the corrected epochs:

| Code name | Paper name | Parameters |
|---|---|---|
| `conformer_b_slim` | LEAN | 25,752 at every window |
| `conformer_a` | EEG Conformer | 153,978 / 169,338 / 201,338 / 221,818 at 800 / 1500 / 3000 / 4000 ms |

Both models, the training loop, the CV splits, the seeds and the hyperparameters are
copied unchanged from `Slim_conformer_fine` and `Conformer_decimated_CWT`. Only the data
is different:

| | September runs (`FineMI_0.5_3hz`) | This project (`fixationcrossbaseline_0.5_3hz_MIonly`) |
|---|---|---|
| Epoch | -0.5 to 4.0 s, 1126 samples | 0 to 4 s, 1000 samples (imagery only) |
| Baseline | -0.5 to 0 s | -4 to -2 s (fixation cross) |
| 4000 ms window | -0.5 to 3.5 s | 0 to 4 s |

Times are relative to the trigger, which marks "Start Imagining". The preflight refuses
any file whose epochs are not exactly 1000 samples, so the old data can't be used by
mistake.

Known data caveat: subject 1 has 360 trials and subject 5 has 321 (the dataset paper
has 320), and channels M1/M2 are kept while CB1/CB2 are dropped. The preflight prints
both trial counts as warnings and doesn't stop.

## Layout

```
run_vast.sh                    the whole run on one Vast.ai instance (smoke | full)
src/config.py                  models, data path, windows, training settings
src/train.py                   training loop (both models)
src/models/                    conformer_a.py, conformer_b_slim.py, blocks.py
scripts/preflight.py           packages, GPU, data (1000-sample epochs), parameter counts,
                               determinism, time projection
scripts/run_selected_pairs.py  trains pairs x models; finished runs are skipped
scripts/check_results.py       every expected file, subject, fold and parameter count
scripts/compare_models.py      LEAN vs EEG Conformer: every number the paper quotes
```

Results: `results/<mode>/<model>/<PAIR>/` and `results/<mode>/comparison/`.

## Step-by-step guide (Vast.ai)

### 0. On your PC: push the code

From `FineMI-enhancements/` in Git Bash:

```bash
git add "Fine Conformer Fixation Cross Fix"
git commit -m "LEAN + EEG Conformer on fixation-baseline, imagery-only data"
git push
```

### 1. Rent an instance

- Image: a PyTorch template with CUDA 12.x and Python 3.10 or newer.
- GPUs: 2-4 x A100 (40 or 80 GB). Each GPU trains its own share of the 28 pairs. The
  models are small, so a cheaper 24 GB card works as well.
- Disk: at least 30 GB (the data is 2.7 GB).
- Note the SSH port and IP from the instance card. Below they are `<PORT>` and `<IP>`.

### 2. On your PC: upload the data

```bash
cd "/c/Users/rmaah/Documents/Summer 2026/Fine MI"
scp -P <PORT> -r fixationcrossbaseline_0.5_3hz_MIonly root@<IP>:/workspace/
```

### 3. On the instance: code and environment

```bash
ssh -p <PORT> root@<IP>
cd /workspace
git clone https://github.com/rmaahin/FineMI-enhancements.git     # or: cd FineMI-enhancements && git pull
python -m venv /workspace/venv --system-site-packages               # reuses the image's torch
source /workspace/venv/bin/activate
pip install numpy==2.0.2 scipy==1.13.1 scikit-learn==1.6.1 pandas==2.3.3
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
ls /workspace/fixationcrossbaseline_0.5_3hz_MIonly | wc -l          # expect 18
```

The pinned versions match the September Vast.ai run.

### 4. Start tmux

The run takes hours, and tmux keeps it alive if SSH drops.

```bash
tmux new -s fix
# detach: Ctrl-b then d      reattach later: tmux attach -t fix
```

### 5. Smoke test (about 10-20 min)

```bash
cd "/workspace/FineMI-enhancements/Fine Conformer Fixation Cross Fix"
bash run_vast.sh smoke
```

It must end with `SMOKE TEST PASSED`. Check two lines in the preflight output:
- `data (all subjects)`: `T=200 @ 50 Hz, was 1000`, with warnings for subjects 1 and 5 only.
- `timing projection (full mode)`: the expected length of the full run on this instance.

### 6. Full run

```bash
bash run_vast.sh full
```

- Progress: `tail -f results/full/logs/gpu0_full_*.log` (one log per GPU).
- If it stops, run `bash run_vast.sh full` again. Finished pairs are skipped.
- It's done when it prints `All runs completed` and `CHECK PASSED`.

### 7. Download the results, then destroy the instance

On the instance, pack them up. This avoids the spaces in the folder name, which `scp`
handles badly:

```bash
tar czf /workspace/fix_results.tgz -C "/workspace/FineMI-enhancements/Fine Conformer Fixation Cross Fix" results
```

On your PC:

```bash
cd "/c/Users/rmaah/Documents/Summer 2026/Fine MI/FineMI-enhancements/Fine Conformer Fixation Cross Fix"
scp -P <PORT> root@<IP>:/workspace/fix_results.tgz .
tar xzf fix_results.tgz
```

Open `results/full/comparison/paper_numbers.md` and confirm the files arrived before you
destroy the instance.

## Updating the paper from `paper_numbers.md`

`paper_numbers.md` gives every number twice: the **window average**, which is what the
September Table 2 and text used even though the caption says "at 4000 ms", and the
**4000 ms** window alone. Pick one and fix the Table 2 caption to match. Then update:

- Abstract: LEAN's average accuracy and the gap to EEG Conformer.
- Contributions: the "retaining X% of its mean accuracy" figure.
- Section 2.1:
  - The trial: "the 4.0 s segment after the 'Start Imagining' cue" (now correct).
  - The baseline: "baseline-corrected on the fixation-cross interval (-4 to -2 s)", which
    replaces "the 0.5 s pre-cue interval".
- Section 3: overall averages, retention %, the average gap, "below 4% in N of the 28
  pairs", the three smallest gaps, the hardest-ten and easiest-ten gaps, Pearson's r,
  and the distant / adjacent / same-joint averages and gaps.
- Table 2: the EEG Conformer and LEAN columns. The rows are ready to paste.
- Conclusion: the three group averages and the overall average.

These don't change: parameter counts, latency figures, and the ShallowConvNet and FINE
columns (those come from other work).
