# ShallowConvNet_8-30

ShallowConvNet (Schirrmeister et al. 2017) on the fixation-cross-baseline, imagery-only
FineMI data, **8-30 Hz**, at the full 250 Hz (no decimation; 50 Hz would alias 8-30 Hz).
The code is identical to `ShallowConvNet_0.5-3` apart from the band block at the top of
`src/config.py`. See that folder's README for the model and training details.

**Before running:** the 8-30 Hz data doesn't exist yet. `src/config.py` assumes it
follows the 0.5-3 Hz naming:

```
Fine MI/fixationcrossbaseline_8_30hz_MIonly/subject{N}_eeg_epochs_8_30hz_250hz.npz
```

That is, the same 0 to 4 s imagery-only epochs (1000 samples at 250 Hz), the same -4 to
-2 s baseline, 62 channels, and `data` / `labels` keys. If the names differ, change
`BAND_TAG` / `DATASET_DIRNAME`. The preflight stops if the epochs aren't 1000 samples or
if most of the power isn't in 8-30 Hz.

`summarize.py` compares the results with `ShallowConvNet_0.5-3/results/<mode>/`, paired
over subjects, so the band comparison is automatic once both have run. Copy the 0.5-3 Hz
results into that folder first if the two runs happen on different instances. If they
are missing, the summary only reports ShallowConvNet's own numbers.

## Vast.ai

The steps are the same as in `ShallowConvNet_0.5-3/README.md`, with this folder and data:

```bash
# PC
cd "/c/Users/rmaah/Documents/Summer 2026/Fine MI"
scp -P <PORT> -r fixationcrossbaseline_8_30hz_MIonly root@<IP>:/workspace/
# instance (after clone / venv / pip as in the 0.5-3 README)
tmux new -s shallow830
cd /workspace/FineMI-enhancements/ShallowConvNet_8-30
bash run_vast.sh smoke                   # must end with SMOKE TEST PASSED
bash run_vast.sh full
# PC, afterwards
cd "/c/Users/rmaah/Documents/Summer 2026/Fine MI/FineMI-enhancements/ShallowConvNet_8-30"
scp -P <PORT> -r root@<IP>:/workspace/FineMI-enhancements/ShallowConvNet_8-30/results .
```

`run_vast.sh` uses every GPU it sees. To do both bands on one instance, run them one
after the other. Upload both data folders and run 0.5-3 first, so the 8-30 summary
finds its results.
