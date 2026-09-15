#!/bin/bash
#SBATCH -J finemi_0.5_3hz
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o results/slurm_%j.out
#SBATCH -e results/slurm_%j.err
# Allocation: add `-A <project>` to the sbatch command if you have more than one.
#
# FINE (base) vs Conformer A vs Conformer B on FineMI 0.5-3 Hz, Lonestar6 gpu-a100-dev.
#
# Submit from Conformer_0.5_3hz/ (results and logs are written relative to it):
#
#   cd $WORK/FineMI-enhancements/Conformer_0.5_3hz
#   export PAIRS="WAA_SAA HOC_WFE EPS_SPS"
#   sbatch submit_ls6_dev.slurm
#
# Settings (environment variables, read at submit time; sbatch exports your env):
#   PAIRS         required. Space-separated: WAA_SAA HOC_WFE ... (or 2_6, or index 0-27)
#   MODE          full (default) | sanity | smoke
#   MODELS        default "fine conformer_a conformer_b"
#   DATASET_ROOT  default $SCRATCH/finemi-dataset/FineMI_0.5_3hz
#   VENV          default $SCRATCH/venvs/fine
#   OVERWRITE     1 = re-train runs that already have results.json (default 0 = skip them)
#
# GPU layout (a gpu-a100 node has 3 A100s):
#   1 pair  -> one architecture per GPU, all three in parallel
#   2+ pairs -> one pair per GPU (round-robin); each worker runs the models in turn
# Previous full runs on this node type took ~22 min/pair (Conformer A) and ~14 min/pair
# (Conformer B); FINE is expected to be <= Conformer B. So 3 pairs ~ 50 min wall time,
# 1 pair ~ 25 min. More than 6 pairs will not fit in the 2 h dev limit. If the job
# times out, just resubmit: finished (pair, model) runs are skipped.

set -euo pipefail

# ---- Settings ------------------------------------------------------------
PAIRS="${PAIRS:-}"
MODE="${MODE:-full}"
MODELS="${MODELS:-fine conformer_a conformer_b}"
DATASET_ROOT="${DATASET_ROOT:-$SCRATCH/finemi-dataset/FineMI_0.5_3hz}"
VENV="${VENV:-$SCRATCH/venvs/fine}"
OVERWRITE="${OVERWRITE:-0}"

if [ -z "$PAIRS" ]; then
    echo "PAIRS is not set. Example: export PAIRS=\"WAA_SAA HOC_WFE EPS_SPS\"; sbatch submit_ls6_dev.slurm" >&2
    exit 1
fi
case "$MODE" in
  smoke|sanity|full) ;;
  *) echo "Unknown MODE=$MODE (expected smoke|sanity|full)" >&2; exit 1 ;;
esac

# ---- Paths ---------------------------------------------------------------
REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"
if [ ! -f scripts/run_selected_pairs.py ]; then
    echo "Run sbatch from the Conformer_0.5_3hz/ directory (got $REPO_ROOT)" >&2
    exit 1
fi

shopt -s nullglob
DATA_FILES=("$DATASET_ROOT"/subject*_eeg_epochs_0.5_3hz_*.npz)
shopt -u nullglob
if [ ${#DATA_FILES[@]} -eq 0 ]; then
    echo "No subject*_eeg_epochs_0.5_3hz_*.npz files in DATASET_ROOT=$DATASET_ROOT" >&2
    exit 1
fi

LOGDIR="results/${MODE}/logs"
mkdir -p "$LOGDIR"

# ---- Environment ---------------------------------------------------------
source "$VENV/bin/activate"
# `python -m scripts.*` from REPO_ROOT already makes `import src.*` resolve;
# PYTHONPATH is set too so it also works if a worker is started another way.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

EXTRA_ARGS=()
if [ "$OVERWRITE" = "1" ]; then
    EXTRA_ARGS+=(--overwrite)
fi

# Validate + normalize pairs (e.g. SAA_WAA -> WAA_SAA, duplicates dropped) before
# splitting them across GPUs, so two workers never train the same pair/model.
NORM_PAIRS=$(python -m scripts.run_selected_pairs --pairs $PAIRS --list-pairs)
read -r -a PAIR_LIST <<< "$NORM_PAIRS"
read -r -a MODEL_LIST <<< "$MODELS"

NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -lt 1 ]; then
    echo "No GPUs visible on $(hostname)" >&2
    exit 1
fi

echo "Job:          $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Partition:    ${SLURM_JOB_PARTITION:-?}"
echo "Mode:         $MODE"
echo "Pairs:        ${PAIR_LIST[*]}"
echo "Models:       ${MODEL_LIST[*]}"
echo "Dataset:      $DATASET_ROOT (${#DATA_FILES[@]} files)"
echo "Results:      $REPO_ROOT/results/$MODE"
echo "GPUs:"
nvidia-smi -L

# ---- Split work across GPUs ----------------------------------------------
GROUP_PAIRS=()
GROUP_MODELS=()
if [ ${#PAIR_LIST[@]} -eq 1 ]; then
    for i in "${!MODEL_LIST[@]}"; do
        g=$((i % NGPU))
        GROUP_PAIRS[g]="${PAIR_LIST[0]}"
        GROUP_MODELS[g]="${GROUP_MODELS[g]:-} ${MODEL_LIST[i]}"
    done
else
    for i in "${!PAIR_LIST[@]}"; do
        g=$((i % NGPU))
        GROUP_PAIRS[g]="${GROUP_PAIRS[g]:-} ${PAIR_LIST[i]}"
        GROUP_MODELS[g]="${MODEL_LIST[*]}"
    done
    PAIRS_PER_GPU=$(( (${#PAIR_LIST[@]} + NGPU - 1) / NGPU ))
    if [ "$MODE" = "full" ] && [ "$PAIRS_PER_GPU" -gt 2 ]; then
        echo "WARNING: $PAIRS_PER_GPU pairs per GPU (~50 min each) will likely exceed the 2 h" \
             "dev limit; resubmit afterwards to finish the remaining runs." >&2
    fi
fi

# ---- Launch one worker per GPU -------------------------------------------
PIDS=()
for g in "${!GROUP_PAIRS[@]}"; do
    LOG="${LOGDIR}/gpu${g}_${SLURM_JOB_ID}.log"
    echo "GPU $g: pairs=[${GROUP_PAIRS[g]# }] models=[${GROUP_MODELS[g]# }] -> $LOG"
    # word splitting of the pair/model groups is intentional
    CUDA_VISIBLE_DEVICES=$g python -m scripts.run_selected_pairs \
        --pairs ${GROUP_PAIRS[g]} \
        --models ${GROUP_MODELS[g]} \
        --mode "$MODE" \
        --dataset-root "$DATASET_ROOT" \
        --no-table --no-log-file ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done
echo "Workers launched: PIDs ${PIDS[*]}"

# Wait for all workers; remember failures but still build the table
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done

# ---- Comparison table (pairs x architectures) ----------------------------
echo ""
echo "Building comparison table..."
python -m scripts.make_comparison_table \
    --pairs "${PAIR_LIST[@]}" --models "${MODEL_LIST[@]}" --mode "$MODE" || FAIL=1

if [ $FAIL -ne 0 ]; then
    echo "One or more workers failed - see ${LOGDIR}/gpu*_${SLURM_JOB_ID}.log" >&2
    exit 1
fi
echo "All runs completed: results/${MODE}/comparison/comparison_table.md"
