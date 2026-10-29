# OFNER experiments — LEAN vs EEG Conformer

This folder compares LEAN (Slim Conformer B) with EEG Conformer (Conformer A) on the Ofner et al. 2017 upper-limb motor-imagery dataset (BNCI 001-2017, Zenodo 834976).

- The task is within-subject binary classification over 15 pairs of the 6 movements: EF, EE, SUP, PRO, HC and HO. Rest trials are not used.
- There are 15 subjects with 360 trials each, recorded on 61 channels.
- It runs in two bands, 0.5–3 Hz and 8–30 Hz.

## Trial timing and windows

| Trial time | Epoch time | Event |
|---|---|---|
| 0 s | −2 s | Beep and fixation cross. The interval from −2 to 0 s is the **baseline only**. |
| 2 s | 0 s | **Cue** (the class trigger), then sustained imagery |
| 5 s | 3 s | End of the 3 s task window (MOABB uses "2–5 s") |

All three windows (800, 1500 and 3000 ms) start at the cue. The fixation period never enters any window. `preprocess_ofner.py` checks from the event codes that the trigger is at the cue.

## Decimation

| Band | Rate the models see | Why |
|---|---|---|
| 0.5–3 Hz | 50 Hz (250 Hz ÷ 5) | Both models were designed at 50 Hz. 3 Hz is far below the 25 Hz Nyquist. |
| 8–30 Hz | 250 Hz (no decimation) | At 50 Hz, the 25–30 Hz content would alias. |

At 250 Hz, the parts of LEAN that act on time are scaled up 5× so they cover the same milliseconds:
- Temporal kernels change from 7/15/31 to 35/75/155 samples.
- The pre-attention pool changes from 3/3 to 15/15.
- The token counts are unchanged.

EEG Conformer uses its original 250 Hz configuration: kernel 25, pool 75/15.

| Parameters | 0.5–3 Hz | 8–30 Hz |
|---|---:|---:|
| LEAN | 25,139 | 38,071 |
| EEG Conformer (800 / 1500 / 3000 ms) | 151,738 / 167,098 / 199,098 | same |

These counts differ from FineMI's because OFNER has 61 channels and its longest window is 3000 ms. The preflight asserts them.

## Layout

```
download/download_ofner.sh                 step 1: 150 GDF files from Zenodo
preprocess/preprocess_ofner.py             step 2: .npz epochs for both bands
preprocess/submit_preprocess_ls6.slurm     step 2 on a TACC CPU node
src/  scripts/                             step 3: models, training, checks, comparison
run_pipeline.sh                            one band: preflight -> train -> check -> compare
run_vast.sh                                Vast.ai wrapper (both bands by default)
submit_ls6.slurm                           TACC ls6 gpu-a100 wrapper (both bands by default)
```

Results go to `results/<band>/<mode>/<model>/<PAIR>/`. The headline numbers are in `results/<band>/full/comparison/ofner_numbers.md`.

## TACC Lonestar6

```bash
cd "$WORK/FineMI-enhancements" && git pull
cd "OFNER experiments"
source $SCRATCH/venvs/fine/bin/activate && pip install mne      # once

# 1. download, on a LOGIN node (compute nodes have no internet) -> $SCRATCH/ofner/raw
bash download/download_ofner.sh

# 2. preprocess -> $SCRATCH/ofner/ofner_mi_0.5_3hz, ofner_mi_8_30hz
SUBJECTS=1,2 sbatch -p development -t 00:30:00 preprocess/submit_preprocess_ls6.slurm
#    read preprocess/slurm_<job>.out (timing check + VERIFICATION table), then:
sbatch preprocess/submit_preprocess_ls6.slurm

# 3. train
mkdir -p results
MODE=smoke sbatch -p gpu-a100-dev -t 01:00:00 submit_ls6.slurm   # both bands
sbatch submit_ls6.slurm                                          # full, both bands
```

The full job runs the 0.5–3 Hz band and then the 8–30 Hz band. Before submitting it, check that the smoke run's timing projection fits in the 12 h limit. The projection is printed per band in `results/<band>/smoke/logs/preflight_*.log`. If the projection doesn't fit, submit one band per job: `BAND=0.5_3hz sbatch submit_ls6.slurm` and `BAND=8_30hz sbatch submit_ls6.slurm`. If a job times out, resubmit it; finished pairs are skipped.

## Vast.ai

```bash
cd /workspace && git clone <repo> && cd "FineMI-enhancements/OFNER experiments"
/workspace/venv/bin/pip install mne

bash download/download_ofner.sh                       # -> /workspace/ofner/raw
python preprocess/preprocess_ofner.py --band both --subjects 1,2   # check first
python preprocess/preprocess_ofner.py --band both

tmux new -s ofner
bash run_vast.sh smoke          # both bands
bash run_vast.sh full           # both bands; or: bash run_vast.sh full 8_30hz
```

## By hand (one band, one GPU)

```bash
python -m scripts.preflight --band 0.5_3hz --project-pairs 15
python -m scripts.run_selected_pairs --band 0.5_3hz --pairs EF_EE SUP_PRO HC_HO --mode smoke
python -m scripts.check_results --band 0.5_3hz --mode smoke --pairs EF_EE SUP_PRO HC_HO
python -m scripts.compare_models --band 0.5_3hz --mode smoke --pairs EF_EE SUP_PRO HC_HO
```

`--dataset-root` overrides the default data folder, which is `$SCRATCH/ofner/ofner_mi_<band>` on TACC and `/workspace/ofner/ofner_mi_<band>` elsewhere.
