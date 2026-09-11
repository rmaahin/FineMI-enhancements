#!/bin/bash
#SBATCH -J finemi
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A YOUR_ALLOCATION_HERE

set -euo pipefail

# ---- Config from env ---------------------------------------------------
: "${MODEL:?set MODEL=conformer_a or conformer_b via sbatch --export}"
MODE="${MODE:-full}"

case "$MODEL" in
  conformer_a) EXPERIMENT=option_a ;;
  conformer_b) EXPERIMENT=option_b ;;
  *) echo "Unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

case "$MODE" in
  smoke|sanity|full) ;;
  *) echo "Unknown MODE=$MODE (expected smoke|sanity|full)" >&2; exit 1 ;;
esac

# Submit from ConformerEEG/ (e.g. cd $WORK/FineMI-enhancements/ConformerEEG)
REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"
if [ ! -f scripts/run_pairs_range.py ]; then
    echo "Run sbatch from the ConformerEEG/ directory (got $REPO_ROOT)" >&2
    exit 1
fi

mkdir -p "experiments/${EXPERIMENT}/logs/${MODE}"
mkdir -p "experiments/${EXPERIMENT}/results/${MODE}"

# ---- Environment -------------------------------------------------------
source "$SCRATCH/venvs/fine/bin/activate"

# `python scripts/*.py` puts scripts/ (not the repo root) on sys.path, so the
# repo root must be on PYTHONPATH for `import src.*` to resolve.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Job:        $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Model:      $MODEL"
echo "Experiment: $EXPERIMENT"
echo "Mode:       $MODE"
echo "GPUs:"
nvidia-smi -L

# ---- Launch 3 concurrent workers, one per GPU --------------------------
LOGDIR="experiments/${EXPERIMENT}/logs/${MODE}"
COMMON="--model $MODEL --mode $MODE --scratch $SCRATCH --repo-root $REPO_ROOT"

CUDA_VISIBLE_DEVICES=0 python scripts/run_pairs_range.py \
    $COMMON --start 0  --end 10 \
    > "${LOGDIR}/gpu0_${SLURM_JOB_ID}.out" \
    2> "${LOGDIR}/gpu0_${SLURM_JOB_ID}.err" &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python scripts/run_pairs_range.py \
    $COMMON --start 10 --end 19 \
    > "${LOGDIR}/gpu1_${SLURM_JOB_ID}.out" \
    2> "${LOGDIR}/gpu1_${SLURM_JOB_ID}.err" &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python scripts/run_pairs_range.py \
    $COMMON --start 19 --end 28 \
    > "${LOGDIR}/gpu2_${SLURM_JOB_ID}.out" \
    2> "${LOGDIR}/gpu2_${SLURM_JOB_ID}.err" &
PID2=$!

echo "Workers launched: PIDs $PID0 $PID1 $PID2"

# Wait for all three; fail the job if any worker exits non-zero
FAIL=0
wait $PID0 || FAIL=1
wait $PID1 || FAIL=1
wait $PID2 || FAIL=1

if [ $FAIL -ne 0 ]; then
    echo "One or more workers failed" >&2
    exit 1
fi

echo "All pairs completed for $MODEL / $MODE"
