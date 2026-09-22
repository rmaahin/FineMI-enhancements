#!/bin/bash
#SBATCH -J slim_conformer_b
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -o results/slurm_%j.out
#SBATCH -e results/slurm_%j.err
# Allocation: add `-A <project>` to the sbatch command if you have more than one.
#
# Slim Conformer B (25,752 parameters) on FineMI 0.5-3 Hz decimated 250 -> 50 Hz, on a
# Lonestar6 A100 node (3 GPUs; no --gres, the job gets the whole node). For the local
# 3-pair test use scripts/local_test.py instead; this script is for the 28-pair sweep.
#
# Submit from Slim_conformer_fine/ (results and logs are written relative to it):
#
#   cd $WORK/FineMI-enhancements/Slim_conformer_fine
#   MODE=smoke sbatch -p gpu-a100-dev -t 00:45:00 submit_ls6.slurm    # smoke test
#   sbatch submit_ls6.slurm                                           # full sweep
#
# Settings (environment variables, read at submit time; sbatch exports your env):
#   MODE             full (default) | smoke
#   PAIRS            full: "all" (default); smoke: "WAA_SAA HOC_WFE EPS_SPS" (default)
#   DATASET_ROOT     default $SCRATCH/finemi-dataset/FineMI_0.5_3hz
#   VENV             default $SCRATCH/venvs/fine
#   OVERWRITE        1 = re-train pairs that already have results (default: 1 smoke, 0 full)
#   FULL_WALLTIME_H  hours the full job may run; the smoke projection must fit in it.
#                    Default 6, keep equal to the "#SBATCH -t" line above.
#
# After training: output check, then the comparison with Conformer A and B from
# ../Conformer_decimated_CWT/results (results/<MODE>/comparison/vs_reference.md).
# Resume: finished pairs are skipped, so a timed-out job is simply resubmitted.

set -euo pipefail

MODEL=conformer_b_slim
MODE="${MODE:-full}"
case "$MODE" in
  smoke)
    PAIRS="${PAIRS:-WAA_SAA HOC_WFE EPS_SPS}"
    OVERWRITE="${OVERWRITE:-1}" ;;
  full)
    PAIRS="${PAIRS:-all}"
    OVERWRITE="${OVERWRITE:-0}" ;;
  *) echo "Unknown MODE=$MODE (expected smoke|full)" >&2; exit 1 ;;
esac
DATASET_ROOT="${DATASET_ROOT:-$SCRATCH/finemi-dataset/FineMI_0.5_3hz}"
VENV="${VENV:-$SCRATCH/venvs/fine}"
FULL_WALLTIME_H="${FULL_WALLTIME_H:-6}"

# ---- Paths ---------------------------------------------------------------
REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"
if [ ! -f scripts/run_selected_pairs.py ] || [ ! -f src/models/conformer_b_slim.py ]; then
    echo "Run sbatch from the Slim_conformer_fine/ directory (got $REPO_ROOT)" >&2
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
if [ ! -f "$VENV/bin/activate" ]; then
    echo "No virtualenv at VENV=$VENV" >&2
    exit 1
fi
source "$VENV/bin/activate"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

EXTRA_ARGS=()
if [ "$OVERWRITE" = "1" ]; then
    EXTRA_ARGS+=(--overwrite)
fi

NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -lt 1 ]; then
    echo "No GPUs visible on $(hostname)" >&2
    exit 1
fi

ALL_PAIRS_STR=$(python -m scripts.run_selected_pairs --pairs $PAIRS --list-pairs --mode "$MODE")
PENDING_STR=$(python -m scripts.run_selected_pairs --pairs $PAIRS --list-pairs --pending \
    --mode "$MODE" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
read -r -a ALL_PAIRS <<< "$ALL_PAIRS_STR"
read -r -a PENDING_PAIRS <<< "$PENDING_STR"

echo "Job:          $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Partition:    ${SLURM_JOB_PARTITION:-?}"
echo "Started:      $(date)"
echo "Mode:         $MODE"
echo "Model:        $MODEL (25,752 parameters)"
echo "Pairs:        ${#ALL_PAIRS[@]} selected, ${#PENDING_PAIRS[@]} still to do"
echo "Pending:      ${PENDING_PAIRS[*]:-(none)}"
echo "Overwrite:    $OVERWRITE"
echo "Dataset:      $DATASET_ROOT (${#DATA_FILES[@]} files)"
echo "Results:      $REPO_ROOT/results/$MODE"
echo "GPUs:"
nvidia-smi -L

# ---- Step 1: preflight ------------------------------------------------------
PREFLIGHT_ARGS=(--dataset-root "$DATASET_ROOT" --mode "$MODE" --pairs "${ALL_PAIRS[@]}"
                --n-gpus "$NGPU" --project-pairs 28 --walltime-hours "$FULL_WALLTIME_H")
if [ "$MODE" = "full" ]; then
    PREFLIGHT_ARGS+=(--skip-timing)
fi
echo ""
echo "Step 1: preflight $(date)"
if ! CUDA_VISIBLE_DEVICES=0 python -m scripts.preflight "${PREFLIGHT_ARGS[@]}" \
        2>&1 | tee "$LOGDIR/preflight_${SLURM_JOB_ID}.log"; then
    echo "PREFLIGHT FAILED - nothing was trained. See $LOGDIR/preflight_${SLURM_JOB_ID}.log" >&2
    exit 1
fi

FAIL=0

# ---- Step 2: training, pairs dealt round-robin over the GPUs -----------------
echo ""
echo "Step 2: training $(date)"
if [ ${#PENDING_PAIRS[@]} -gt 0 ]; then
    GROUP_PAIRS=()
    for i in "${!PENDING_PAIRS[@]}"; do
        g=$((i % NGPU))
        GROUP_PAIRS[g]="${GROUP_PAIRS[g]:-} ${PENDING_PAIRS[i]}"
    done
    PIDS=()
    for g in "${!GROUP_PAIRS[@]}"; do
        LOG="${LOGDIR}/gpu${g}_${SLURM_JOB_ID}.log"
        echo "GPU $g: pairs=[${GROUP_PAIRS[g]# }] -> $LOG"
        # word splitting of the pair group is intentional
        CUDA_VISIBLE_DEVICES=$g python -m scripts.run_selected_pairs \
            --pairs ${GROUP_PAIRS[g]} \
            --mode "$MODE" \
            --dataset-root "$DATASET_ROOT" \
            --no-log-file ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
            > "$LOG" 2>&1 &
        PIDS+=($!)
    done
    echo "Workers launched: PIDs ${PIDS[*]}"
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAIL=1
    done
    echo "Workers finished: $(date)"
else
    echo "Nothing to train: every selected pair is complete."
fi

# ---- Step 3: output check + comparison ----------------------------------------
echo ""
echo "Step 3: checking outputs $(date)"
python -m scripts.check_results --mode "$MODE" --pairs "${ALL_PAIRS[@]}" \
    --dataset-root "$DATASET_ROOT" || FAIL=1

echo ""
echo "Step 4: comparison with Conformer A and B $(date)"
python -m scripts.compare_to_reference --mode "$MODE" --pairs "${ALL_PAIRS[@]}" || FAIL=1

if [ "$MODE" = "smoke" ]; then
    LEFT=$(python -m scripts.run_selected_pairs --pairs "${ALL_PAIRS[@]}" --list-pairs \
        --pending --mode smoke)
    if [ -n "$LEFT" ]; then
        echo "Resume check FAILED: still listed as pending after training: $LEFT" >&2
        FAIL=1
    else
        echo "Resume check passed: finished runs are recognized and would be skipped."
    fi
fi

echo ""
if [ $FAIL -ne 0 ]; then
    echo "One or more steps failed - see ${LOGDIR}/*_${SLURM_JOB_ID}.log and results/slurm_${SLURM_JOB_ID}.err" >&2
    [ "$MODE" = "smoke" ] && echo "SMOKE TEST FAILED - do not start the full sweep yet." >&2
    exit 1
fi
if [ "$MODE" = "smoke" ]; then
    echo "SMOKE TEST PASSED $(date)"
    echo "Next: sbatch submit_ls6.slurm   (from $REPO_ROOT, with MODE and PAIRS unset)"
else
    echo "All runs completed $(date): results/full/comparison/vs_reference.md"
fi
