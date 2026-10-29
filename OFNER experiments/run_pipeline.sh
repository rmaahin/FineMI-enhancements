#!/bin/bash
# LEAN (conformer_b_slim) + EEG Conformer (conformer_a) on ONE OFNER band, on every GPU
# this machine can see. Shared by run_vast.sh (Vast.ai) and submit_ls6.slurm (TACC);
# call those, not this, unless you set the environment yourself.
#
#   bash run_pipeline.sh smoke|full 0.5_3hz|8_30hz
#
# Steps: preflight -> training (pairs dealt round-robin over the GPUs, each worker
# trains every model for its pairs) -> output check -> LEAN vs EEG Conformer numbers.
#
# Settings (environment variables):
#   DATASET_ROOT     default $SCRATCH/ofner/ofner_mi_<band> if $SCRATCH is set,
#                    else /workspace/ofner/ofner_mi_<band>
#   VENV             virtualenv to activate (skipped if unset or absent)
#   PAIRS            smoke: "EF_EE SUP_PRO HC_HO" (default); full: "all" (default)
#   MODELS           default "conformer_b_slim conformer_a"
#   OVERWRITE        1 = re-train finished runs (default: 1 smoke, 0 full)
#   FULL_WALLTIME_H  smoke only: fail if the projected full run exceeds this many hours
#                    (default 0 = just report the projection)
#   RUN_TAG          log-file tag (default <mode>_<band>_<timestamp>)
#
# Resume: finished (pair, model) runs are skipped, so after an interruption just run
# the same command again.
#
# Output: results/<band>/<mode>/<model>/<PAIR>/,
#         results/<band>/<mode>/comparison/ofner_numbers.md,
#         results/<band>/<mode>/logs/ (preflight, one log per GPU, this script's own log)

set -euo pipefail

MODE="${1:-}"
BAND_IN="${2:-}"
case "$MODE" in
  smoke)
    PAIRS="${PAIRS:-EF_EE SUP_PRO HC_HO}"
    OVERWRITE="${OVERWRITE:-1}" ;;
  full)
    PAIRS="${PAIRS:-all}"
    OVERWRITE="${OVERWRITE:-0}" ;;
  *) echo "Usage: bash run_pipeline.sh smoke|full 0.5_3hz|8_30hz" >&2; exit 1 ;;
esac
# 0.5_3, 0.5-3, 0.5_3hz, 8_30 ... -> 0.5_3hz / 8_30hz
BAND="${BAND_IN//-/_}"
BAND="${BAND%hz}hz"
case "$BAND" in
  0.5_3hz|8_30hz) ;;
  *) echo "Unknown band '$BAND_IN' (expected 0.5_3hz or 8_30hz)" >&2; exit 1 ;;
esac
MODELS="${MODELS:-conformer_b_slim conformer_a}"
if [ -n "${SCRATCH:-}" ]; then
    DATASET_ROOT="${DATASET_ROOT:-$SCRATCH/ofner/ofner_mi_$BAND}"
else
    DATASET_ROOT="${DATASET_ROOT:-/workspace/ofner/ofner_mi_$BAND}"
fi
VENV="${VENV:-}"
FULL_WALLTIME_H="${FULL_WALLTIME_H:-0}"

# ---- Paths -------------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if [ ! -f scripts/run_selected_pairs.py ] || [ ! -f src/models/conformer_a.py ]; then
    echo "run_pipeline.sh must sit in the 'OFNER experiments' folder (got $PROJECT_DIR)" >&2
    exit 1
fi

shopt -s nullglob
DATA_FILES=("$DATASET_ROOT"/subject*_eeg_epochs_"$BAND"_*.npz)
shopt -u nullglob
if [ ${#DATA_FILES[@]} -eq 0 ]; then
    echo "No subject*_eeg_epochs_${BAND}_*.npz files in DATASET_ROOT=$DATASET_ROOT" >&2
    echo "Run preprocess/preprocess_ofner.py first." >&2
    exit 1
fi

RUN_TAG="${RUN_TAG:-${MODE}_${BAND}_$(date +%Y%m%d_%H%M%S)}"
export RUN_TAG
LOGDIR="results/${BAND}/${MODE}/logs"
mkdir -p "$LOGDIR"
# Everything this script prints also goes to its own log.
exec > >(tee -a "$LOGDIR/pipeline_${RUN_TAG}.out") 2>&1

# ---- Environment -------------------------------------------------------------
if [ -n "$VENV" ] && [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
else
    echo "Note: no virtualenv at VENV=${VENV:-(unset)}, using $(command -v python)"
fi
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

EXTRA_ARGS=()
if [ "$OVERWRITE" = "1" ]; then
    EXTRA_ARGS+=(--overwrite)
fi
read -r -a MODEL_LIST <<< "$MODELS"
COMMON=(--band "$BAND" --mode "$MODE" --models "${MODEL_LIST[@]}")

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l || true)
if [ "$NGPU" -lt 1 ]; then
    echo "No GPUs visible (nvidia-smi -L lists none)" >&2
    exit 1
fi

# Normalize pairs ("all" -> 15); PENDING drops pairs already complete for every model.
ALL_PAIRS_STR=$(python -m scripts.run_selected_pairs --pairs $PAIRS --list-pairs "${COMMON[@]}")
PENDING_STR=$(python -m scripts.run_selected_pairs --pairs $PAIRS --list-pairs --pending \
    "${COMMON[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
read -r -a ALL_PAIRS <<< "$ALL_PAIRS_STR"
read -r -a PENDING_PAIRS <<< "$PENDING_STR"

echo "Run:          $RUN_TAG"
echo "Host:         $(hostname)"
echo "Started:      $(date)"
echo "Mode:         $MODE"
echo "Band:         $BAND"
echo "Models:       ${MODEL_LIST[*]}"
echo "Pairs:        ${#ALL_PAIRS[@]} selected, ${#PENDING_PAIRS[@]} with runs still to do"
echo "Pending:      ${PENDING_PAIRS[*]:-(none)}"
echo "Overwrite:    $OVERWRITE"
echo "Dataset:      $DATASET_ROOT (${#DATA_FILES[@]} files)"
echo "Results:      $PROJECT_DIR/results/$BAND/$MODE"
echo "GPUs:"
nvidia-smi -L

# ---- Step 1: preflight (stops before any training if it fails) ------------------
PREFLIGHT_ARGS=(--dataset-root "$DATASET_ROOT" "${COMMON[@]}"
                --n-gpus "$NGPU" --project-pairs 15 --walltime-hours "$FULL_WALLTIME_H")
if [ "$MODE" = "full" ]; then
    PREFLIGHT_ARGS+=(--skip-timing)
fi
echo ""
echo "Step 1: preflight $(date)"
if ! CUDA_VISIBLE_DEVICES=0 python -m scripts.preflight "${PREFLIGHT_ARGS[@]}" \
        > "$LOGDIR/preflight_${RUN_TAG}.log" 2>&1; then
    cat "$LOGDIR/preflight_${RUN_TAG}.log"
    echo "PREFLIGHT FAILED - nothing was trained. See $LOGDIR/preflight_${RUN_TAG}.log" >&2
    exit 1
fi
cat "$LOGDIR/preflight_${RUN_TAG}.log"

FAIL=0

# ---- Step 2: training, pairs dealt round-robin over the GPUs ---------------------
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
        LOG="${LOGDIR}/gpu${g}_${RUN_TAG}.log"
        echo "GPU $g: pairs=[${GROUP_PAIRS[g]# }] -> $LOG"
        # word splitting of the pair group is intentional
        CUDA_VISIBLE_DEVICES=$g python -m scripts.run_selected_pairs \
            --pairs ${GROUP_PAIRS[g]} \
            "${COMMON[@]}" \
            --dataset-root "$DATASET_ROOT" \
            --no-log-file ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
            > "$LOG" 2>&1 &
        PIDS+=($!)
    done
    echo "Workers launched: PIDs ${PIDS[*]}  (follow one with: tail -f \"${LOGDIR}/gpu0_${RUN_TAG}.log\")"
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAIL=1
    done
    echo "Workers finished: $(date)"
    if [ $FAIL -ne 0 ]; then
        echo "A worker failed. Its traceback is in:" >&2
        grep -l "FAILED\|Traceback" "${LOGDIR}"/gpu*_"${RUN_TAG}".log >&2 || true
    fi
else
    echo "Nothing to train: every selected pair is complete for every model."
fi

# ---- Step 3: output check -----------------------------------------------------------
echo ""
echo "Step 3: checking outputs $(date)"
python -m scripts.check_results "${COMMON[@]}" --pairs "${ALL_PAIRS[@]}" \
    --dataset-root "$DATASET_ROOT" || FAIL=1

# ---- Step 4: LEAN vs EEG Conformer ----------------------------------------------------
if [ ${#MODEL_LIST[@]} -eq 2 ]; then
    echo ""
    echo "Step 4: LEAN vs EEG Conformer $(date)"
    python -m scripts.compare_models --band "$BAND" --mode "$MODE" --pairs "${ALL_PAIRS[@]}" \
        || FAIL=1
fi

if [ "$MODE" = "smoke" ]; then
    LEFT=$(python -m scripts.run_selected_pairs --pairs "${ALL_PAIRS[@]}" --list-pairs \
        --pending "${COMMON[@]}")
    if [ -n "$LEFT" ]; then
        echo "Resume check FAILED: still listed as pending after training: $LEFT" >&2
        FAIL=1
    else
        echo "Resume check passed: finished runs are recognized and would be skipped."
    fi
fi

echo ""
if [ $FAIL -ne 0 ]; then
    echo "One or more steps failed ($BAND) - see ${LOGDIR}/*_${RUN_TAG}.log" >&2
    [ "$MODE" = "smoke" ] && echo "SMOKE TEST FAILED ($BAND) - do not start the full run yet." >&2
    exit 1
fi
if [ "$MODE" = "smoke" ]; then
    echo "SMOKE TEST PASSED ($BAND) $(date)"
    echo "Full-run projection: see 'timing projection' in $LOGDIR/preflight_${RUN_TAG}.log"
else
    echo "All runs completed ($BAND) $(date): results/$BAND/full/comparison/ofner_numbers.md"
fi
