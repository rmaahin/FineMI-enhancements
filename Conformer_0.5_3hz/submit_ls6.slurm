#!/bin/bash
#SBATCH -J finemi_0.5_3hz
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 16:00:00
#SBATCH -o results/slurm_%j.out
#SBATCH -e results/slurm_%j.err
# Allocation: add `-A <project>` to the sbatch command if you have more than one.
#
# FINE (base) vs Conformer A vs Conformer B on FineMI 0.5-3 Hz, Lonestar6 A100 node
# (3 GPUs). Default: full sweep of all 28 pairs x 3 architectures x 4 windows.
#
# Submit from Conformer_0.5_3hz/ (results and logs are written relative to it):
#
#   cd $WORK/FineMI-enhancements/Conformer_0.5_3hz
#   sbatch submit_ls6.slurm                                   # all 28 pairs, gpu-a100
#
#   export PAIRS="WAA_SAA HOC_WFE EPS_SPS"                    # a subset instead
#   sbatch -p gpu-a100-dev -t 02:00:00 submit_ls6.slurm       # e.g. on the dev queue
#
# Settings (environment variables, read at submit time; sbatch exports your env):
#   PAIRS         "all" (default) or space-separated: WAA_SAA HOC_WFE ... (or 2_6, or index 0-27)
#   MODE          full (default) | sanity | smoke
#   MODELS        default "fine conformer_a conformer_b"
#   DATASET_ROOT  default $SCRATCH/finemi-dataset/FineMI_0.5_3hz
#   VENV          default $SCRATCH/venvs/fine
#   OVERWRITE     1 = re-train runs that already have results.json (default 0 = skip them)
#
# Resume: (pair, model) runs with a results.json are skipped, and pairs that are
# already complete for every model are dropped before the work is split across
# GPUs. If the job times out or a worker dies, just resubmit the same command.
#
# GPU layout:
#   1 pending pair  -> one architecture per GPU, in parallel
#   2+ pending pairs -> pairs dealt round-robin across the GPUs; each worker trains
#                       FINE, Conformer A, Conformer B for its pairs in turn
# Timing on this node type: Conformer A ~22 min/pair, Conformer B ~14 min/pair,
# FINE assumed <= B, so ~50 min per pair per GPU. 25 pending pairs on 3 GPUs
# (9/8/8) is ~7.5 h; 28 pairs (10/9/9) ~8.5 h. 16 h leaves margin.
#
# After training: comparison table (scripts/make_comparison_table.py) and paired
# tests (scripts/paired_stats.py) over the selected pairs, written to
# results/<MODE>/comparison/.

set -euo pipefail

# ---- Settings ------------------------------------------------------------
PAIRS="${PAIRS:-all}"
MODE="${MODE:-full}"
MODELS="${MODELS:-fine conformer_a conformer_b}"
DATASET_ROOT="${DATASET_ROOT:-$SCRATCH/finemi-dataset/FineMI_0.5_3hz}"
VENV="${VENV:-$SCRATCH/venvs/fine}"
OVERWRITE="${OVERWRITE:-0}"
MIN_PER_PAIR=50   # rough per-GPU minutes for one pair x 3 architectures (full mode)

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

read -r -a MODEL_LIST <<< "$MODELS"

# Validate + normalize pairs ("all" -> 28, SAA_WAA -> WAA_SAA, duplicates dropped).
# ALL_PAIRS is everything selected (for the tables); PENDING_PAIRS drops pairs that
# are already complete for every model, so GPUs aren't handed finished work.
ALL_PAIRS_STR=$(python -m scripts.run_selected_pairs --pairs $PAIRS --list-pairs \
    --models "${MODEL_LIST[@]}" --mode "$MODE")
PENDING_STR=$(python -m scripts.run_selected_pairs --pairs $PAIRS --list-pairs --pending \
    --models "${MODEL_LIST[@]}" --mode "$MODE" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
read -r -a ALL_PAIRS <<< "$ALL_PAIRS_STR"
read -r -a PENDING_PAIRS <<< "$PENDING_STR"

NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -lt 1 ]; then
    echo "No GPUs visible on $(hostname)" >&2
    exit 1
fi

echo "Job:          $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Partition:    ${SLURM_JOB_PARTITION:-?}"
echo "Started:      $(date)"
echo "Mode:         $MODE"
echo "Models:       ${MODEL_LIST[*]}"
echo "Pairs:        ${#ALL_PAIRS[@]} selected, ${#PENDING_PAIRS[@]} with runs still to do"
echo "Pending:      ${PENDING_PAIRS[*]:-(none)}"
echo "Dataset:      $DATASET_ROOT (${#DATA_FILES[@]} files)"
echo "Results:      $REPO_ROOT/results/$MODE"
echo "GPUs:"
nvidia-smi -L

FAIL=0

if [ ${#PENDING_PAIRS[@]} -gt 0 ]; then
    # ---- Split pending work across GPUs ----------------------------------
    GROUP_PAIRS=()
    GROUP_MODELS=()
    if [ ${#PENDING_PAIRS[@]} -eq 1 ]; then
        for i in "${!MODEL_LIST[@]}"; do
            g=$((i % NGPU))
            GROUP_PAIRS[g]="${PENDING_PAIRS[0]}"
            GROUP_MODELS[g]="${GROUP_MODELS[g]:-} ${MODEL_LIST[i]}"
        done
    else
        for i in "${!PENDING_PAIRS[@]}"; do
            g=$((i % NGPU))
            GROUP_PAIRS[g]="${GROUP_PAIRS[g]:-} ${PENDING_PAIRS[i]}"
            GROUP_MODELS[g]="${MODEL_LIST[*]}"
        done
        PAIRS_PER_GPU=$(( (${#PENDING_PAIRS[@]} + NGPU - 1) / NGPU ))
        if [ "$MODE" = "full" ]; then
            echo "Estimate:     ~$(( PAIRS_PER_GPU * MIN_PER_PAIR / 60 )) h" \
                 "($PAIRS_PER_GPU pairs on the busiest GPU x ~${MIN_PER_PAIR} min)"
            if [ "${SLURM_JOB_PARTITION:-}" = "gpu-a100-dev" ] && [ "$PAIRS_PER_GPU" -gt 2 ]; then
                echo "WARNING: $PAIRS_PER_GPU pairs per GPU will likely exceed the 2 h dev" \
                     "limit; resubmit afterwards to finish the remaining runs." >&2
            fi
        fi
    fi

    # ---- Launch one worker per GPU ---------------------------------------
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

    # Wait for all workers; remember failures but still build the tables
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAIL=1
    done
    echo "Workers finished: $(date)"
else
    echo "Nothing to train: every selected pair is complete for every model."
fi

# ---- Comparison table + paired tests (pairs x architectures) ---------------
echo ""
echo "Building comparison table..."
python -m scripts.make_comparison_table \
    --pairs "${ALL_PAIRS[@]}" --models "${MODEL_LIST[@]}" --mode "$MODE" || FAIL=1

if [ ${#MODEL_LIST[@]} -gt 1 ]; then
    echo ""
    echo "Computing paired statistics..."
    python -m scripts.paired_stats \
        --pairs "${ALL_PAIRS[@]}" --models "${MODEL_LIST[@]}" --mode "$MODE" || FAIL=1
fi

if [ $FAIL -ne 0 ]; then
    echo "One or more steps failed - see ${LOGDIR}/gpu*_${SLURM_JOB_ID}.log and results/slurm_${SLURM_JOB_ID}.err" >&2
    exit 1
fi
echo "All runs completed: results/${MODE}/comparison/comparison_table.md and paired_stats.md"
