#!/bin/bash
#SBATCH -J cwt_mod_0.5_3hz
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o slurm_%j.out
#SBATCH -e slurm_%j.err
# Allocation: add `-A <project>` to the sbatch command if you have more than one.
#
# CWT-modified: FINE-raw vs FINE-CWT on FineMI 0.5-3 Hz, decimated 250 -> 50 Hz (D=5),
# on a Lonestar6 A100 node (3 GPUs, no --gres needed: the job gets the whole node).
# Default: all 28 pairs x 2 arms, one worker per GPU, then aggregate.
#
# Submit from CWT-modified/ (results and logs are written relative to it):
#
#   cd $WORK/FineMI-enhancements/CWT-modified
#   sbatch submit_ls6.slurm                                     # full sweep, gpu-a100
#
#   MODE=selftest sbatch -p gpu-a100-dev -t 00:15:00 submit_ls6.slurm   # checks only
#   MODE=smoke PAIRS="HOC_SPS" sbatch -p gpu-a100-dev -t 00:30:00 submit_ls6.slurm
#   PAIRS="HOC_SPS WAA_SAA EPS_SPS" sbatch -p gpu-a100-dev -t 02:00:00 submit_ls6.slurm
#
# Settings (environment variables, read at submit time; sbatch exports your env):
#   PAIRS         "all" (default) or space-separated: HOC_SPS WAA_SAA ... (or 0-5, or HOC/SPS)
#   MODE          full (default) | smoke | selftest
#   ARMS          default "raw cwt"  (either one alone is fine)
#   DATASET_ROOT  default $SCRATCH/finemi-dataset/FineMI_0.5_3hz
#   VENV          default $SCRATCH/venvs/fine
#   SHARDS        workers per arm, default = number of GPUs on the node (3)
#   REDO          1 = re-train pairs that already have results (default 0 = skip them)
#   FINE_BEST_CKPT  1 = deepcopy best-val checkpoint; default 0 = original final-epoch rule
#
# Resume: finished pairs (results.json with matching pipeline stamp) are skipped, so if
# the job times out just resubmit the same command.
#
# Timing (previous 8-30 Hz sweep on this node type): raw ~3 min/pair, cwt ~6 min/pair
# per GPU. Decimation shrinks the raw arm 5x in time, so expect ~1 h for the full sweep
# with 3 workers; 4 h walltime leaves plenty of margin.

set -euo pipefail

# ---- Settings ------------------------------------------------------------
PAIRS="${PAIRS:-all}"
MODE="${MODE:-full}"
ARMS="${ARMS:-raw cwt}"
DATASET_ROOT="${DATASET_ROOT:-$SCRATCH/finemi-dataset/FineMI_0.5_3hz}"
VENV="${VENV:-$SCRATCH/venvs/fine}"
REDO="${REDO:-0}"

case "$MODE" in
  smoke|full|selftest) ;;
  *) echo "Unknown MODE=$MODE (expected smoke|full|selftest)" >&2; exit 1 ;;
esac

# ---- Paths ---------------------------------------------------------------
REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"
if [ ! -f fine_mi.py ] || [ ! -f run_local.py ]; then
    echo "Run sbatch from the CWT-modified/ directory (got $REPO_ROOT)" >&2
    exit 1
fi

shopt -s nullglob
DATA_FILES=("$DATASET_ROOT"/subject*_eeg_epochs_0.5_3hz_*.npz)
shopt -u nullglob
if [ ${#DATA_FILES[@]} -eq 0 ]; then
    echo "No subject*_eeg_epochs_0.5_3hz_*.npz files in DATASET_ROOT=$DATASET_ROOT" >&2
    exit 1
fi

export FINE_DATASET_ROOT="$DATASET_ROOT"
export FINE_RESULTS_ROOT="$REPO_ROOT/results"
LOGDIR="$FINE_RESULTS_ROOT/logs"
mkdir -p "$LOGDIR"

# ---- Environment ---------------------------------------------------------
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -lt 1 ]; then
    echo "No GPUs visible on $(hostname)" >&2
    exit 1
fi
SHARDS="${SHARDS:-$NGPU}"

EXTRA_ARGS=()
[ "$REDO" = "1" ] && EXTRA_ARGS+=(--redo)
[ "$MODE" = "smoke" ] && EXTRA_ARGS+=(--smoke)

echo "Job:          $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Partition:    ${SLURM_JOB_PARTITION:-?}"
echo "Started:      $(date)"
echo "Mode:         $MODE"
echo "Arms:         $ARMS"
echo "Pairs:        $PAIRS"
echo "Shards:       $SHARDS (on $NGPU GPUs)"
echo "Dataset:      $DATASET_ROOT (${#DATA_FILES[@]} files)"
echo "Results:      $FINE_RESULTS_ROOT"
echo "Best ckpt:    ${FINE_BEST_CKPT:-0}"
echo "GPUs:"
nvidia-smi -L

# ---- Self-test (always; it is fast and catches a bad venv/data path early) --
echo ""
echo "Self-test..."
CUDA_VISIBLE_DEVICES=0 python run_local.py --selftest 2>&1 | tee "$LOGDIR/selftest_${SLURM_JOB_ID}.log"
[ "$MODE" = "selftest" ] && { echo "selftest only - done $(date)"; exit 0; }

# ---- Sweep: one worker per shard, per arm ----------------------------------
FAIL=0
for arm in $ARMS; do
    echo ""
    echo "ARM '$arm' starting $(date) ($SHARDS shard(s))"
    PIDS=()
    for k in $(seq 1 "$SHARDS"); do
        g=$(( (k - 1) % NGPU ))
        LOG="${LOGDIR}/${arm}_shard${k}_${SLURM_JOB_ID}.log"
        echo "  shard $k/$SHARDS -> GPU $g -> $LOG"
        CUDA_VISIBLE_DEVICES=$g python run_local.py \
            --mode "$arm" --pairs "$PAIRS" --shard "$k/$SHARDS" \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
            > "$LOG" 2>&1 &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAIL=1
    done
    echo "ARM '$arm' finished $(date)"
done

# ---- Aggregate -------------------------------------------------------------
if [ "$MODE" = "full" ]; then
    echo ""
    echo "Aggregating..."
    python aggregate.py --results-root "$FINE_RESULTS_ROOT" 2>&1 \
        | tee "$LOGDIR/aggregate_${SLURM_JOB_ID}.log" || FAIL=1
fi

if [ $FAIL -ne 0 ]; then
    echo "One or more workers failed - see ${LOGDIR}/*_${SLURM_JOB_ID}.log and results/run_log_shard*.txt" >&2
    exit 1
fi
echo "All done $(date): results/comparison_table.csv, per_pair_summary.csv, per_subject_summary.csv"
