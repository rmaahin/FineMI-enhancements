#!/bin/bash
#SBATCH -J ofner_preprocess
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o preprocess/slurm_%j.out
#SBATCH -e preprocess/slurm_%j.err
# Allocation: add `-A <project>` to the sbatch command if you have more than one.
#
# Epoch the OFNER motor-imagery GDF files into both bands (0.5-3 Hz and 8-30 Hz) on a
# Lonestar6 CPU node. Download first, on a LOGIN node (compute nodes have no internet):
#
#   cd "$WORK/FineMI-enhancements/OFNER experiments"
#   bash download/download_ofner.sh
#   SUBJECTS=1,2 sbatch -p development -t 00:30:00 preprocess/submit_preprocess_ls6.slurm  # check
#   sbatch preprocess/submit_preprocess_ls6.slurm                                          # all 15
#
# Submit from "OFNER experiments/" (the -o/-e paths above are relative to it).
#
# Settings (environment variables, read at submit time; sbatch exports your env):
#   DATA_ROOT  default $SCRATCH/ofner   (raw/ in, ofner_mi_<band>/ out)
#   VENV       default $SCRATCH/venvs/fine   (needs mne: pip install mne)
#   BAND       both (default) | 0.5_3 | 8_30
#   SUBJECTS   default all, e.g. 1,2
#   FORCE      1 = recompute subjects already written

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$SCRATCH/ofner}"
VENV="${VENV:-$SCRATCH/venvs/fine}"
BAND="${BAND:-both}"
SUBJECTS="${SUBJECTS:-}"
FORCE="${FORCE:-0}"

cd "$SLURM_SUBMIT_DIR"
if [ ! -f preprocess/preprocess_ofner.py ]; then
    echo "Run sbatch from the 'OFNER experiments' directory (got $SLURM_SUBMIT_DIR)" >&2
    exit 1
fi
if [ ! -f "$VENV/bin/activate" ]; then
    echo "No virtualenv at VENV=$VENV" >&2
    exit 1
fi
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

ARGS=(--raw "$DATA_ROOT/raw" --out-root "$DATA_ROOT" --band "$BAND")
[ -n "$SUBJECTS" ] && ARGS+=(--subjects "$SUBJECTS")
[ "$FORCE" = "1" ] && ARGS+=(--force)

echo "Job:      $SLURM_JOB_ID on $(hostname), started $(date)"
echo "Command:  python preprocess/preprocess_ofner.py ${ARGS[*]}"
python preprocess/preprocess_ofner.py "${ARGS[@]}"
echo "Finished $(date)"
