#!/bin/bash
# LEAN_v2: LEAN rebuilt step by step (ladder R0-R7), leave-one-out ablations, training-
# only changes and EEGNet baselines, on the fixation-cross-baseline, imagery-only
# 0.5-3 Hz FineMI data at 50 Hz. One Vast.ai instance, one or more GPUs, plain bash.
#
#   bash run_vast.sh smoke     # ~15-30 min: preflight (every preset) + timing projection,
#                              # all presets on 3 pairs x 2 subjects x 2 folds x 2 epochs
#   bash run_vast.sh screen    # 6 pairs (2 distant, 2 adjacent, 2 same-joint), 800 and
#                              # 4000 ms, all 18 subjects, 5 folds; default MODELS=ladder
#   MODELS="lean_v2_r7 eegnet_50_recipe" bash run_vast.sh full    # 28 pairs, 4 windows
#
# Run it inside tmux so it survives an SSH disconnect (see README.md).
#
# Settings (environment variables):
#   MODELS           presets and/or groups: ladder, loo, training, baselines, all
#                    (default: smoke all, screen ladder; full has no default - name them)
#   DATASET_ROOT     default /workspace/fixationcrossbaseline_0.5_3hz_MIonly
#   VENV             virtualenv to activate (default /workspace/venv; skipped if absent)
#   PAIRS            default: smoke 3 pairs, screen the 6 screen pairs, full all 28
#   WORKERS_PER_GPU  training processes per GPU (default 1; 3 is a good start on a
#                    24 GB card - these models are tiny; ~1.5 GB RAM per worker)
#   OVERWRITE        1 = re-train finished runs (default: 1 smoke, 0 otherwise)
#   FULL_WALLTIME_H  smoke only: fail if the projected full run exceeds this many hours
#
# Resume: finished (pair, preset) runs are skipped, so after an interruption just run
# the same command again. Editing a preset in src/config.py changes its hash, so its
# old results are re-trained automatically.
#
# Output: results/<mode>/<preset>/<PAIR>/, results/<mode>/summary/summary.md,
#         results/<mode>/logs/

set -euo pipefail

MODE="${1:-${MODE:-}}"
case "$MODE" in
  smoke)
    MODELS="${MODELS:-all}"
    OVERWRITE="${OVERWRITE:-1}" ;;
  screen)
    MODELS="${MODELS:-ladder}"
    OVERWRITE="${OVERWRITE:-0}" ;;
  full)
    if [ -z "${MODELS:-}" ]; then
        echo "full mode: name the presets, e.g. MODELS=\"lean_v2_r7 eegnet_50_recipe\" bash run_vast.sh full" >&2
        exit 1
    fi
    OVERWRITE="${OVERWRITE:-0}" ;;
  *) echo "Usage: bash run_vast.sh smoke|screen|full" >&2; exit 1 ;;
esac
PAIRS="${PAIRS:-}"
VENV="${VENV:-/workspace/venv}"
FULL_WALLTIME_H="${FULL_WALLTIME_H:-0}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
if ! [[ "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "WORKERS_PER_GPU must be a positive integer (got '$WORKERS_PER_GPU')" >&2
    exit 1
fi

# ---- Paths -------------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if [ ! -f scripts/run_selected_pairs.py ] || [ ! -f src/models/lean_v2.py ]; then
    echo "run_vast.sh must sit in the LEAN_v2 folder (got $PROJECT_DIR)" >&2
    exit 1
fi

RUN_TAG="${MODE}_$(date +%Y%m%d_%H%M%S)"
export RUN_TAG
LOGDIR="results/${MODE}/logs"
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/run_vast_${RUN_TAG}.out") 2>&1

# ---- Environment -------------------------------------------------------------
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
else
    echo "Note: no virtualenv at VENV=$VENV, using $(command -v python)"
fi
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

BAND_TAG=$(python -c "from src.config import BAND_TAG; print(BAND_TAG)")
DATASET_DIRNAME=$(python -c "from src.config import DATASET_DIRNAME; print(DATASET_DIRNAME)")
DATASET_ROOT="${DATASET_ROOT:-/workspace/$DATASET_DIRNAME}"

shopt -s nullglob
DATA_FILES=("$DATASET_ROOT"/subject*_eeg_epochs_"${BAND_TAG}"_*.npz)
shopt -u nullglob
if [ ${#DATA_FILES[@]} -eq 0 ]; then
    echo "No subject*_eeg_epochs_${BAND_TAG}_*.npz files in DATASET_ROOT=$DATASET_ROOT" >&2
    exit 1
fi

EXTRA_ARGS=()
if [ "$OVERWRITE" = "1" ]; then
    EXTRA_ARGS+=(--overwrite)
fi
PAIR_ARGS=()
if [ -n "$PAIRS" ]; then
    read -r -a PAIR_LIST <<< "$PAIRS"
    PAIR_ARGS=(--pairs "${PAIR_LIST[@]}")
fi

# Expand groups (ladder, loo, ...) into preset names
MODELS_STR=$(python -m scripts.run_selected_pairs --models $MODELS --list-models)
read -r -a MODEL_LIST <<< "$MODELS_STR"

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l || true)
if [ "$NGPU" -lt 1 ]; then
    echo "No GPUs visible (nvidia-smi -L lists none)" >&2
    exit 1
fi
NSLOTS=$((NGPU * WORKERS_PER_GPU))

ALL_PAIRS_STR=$(python -m scripts.run_selected_pairs --mode "$MODE" ${PAIR_ARGS[@]+"${PAIR_ARGS[@]}"} \
    --models "${MODEL_LIST[@]}" --list-pairs)
PENDING_STR=$(python -m scripts.run_selected_pairs --mode "$MODE" ${PAIR_ARGS[@]+"${PAIR_ARGS[@]}"} \
    --models "${MODEL_LIST[@]}" --list-pairs --pending ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
read -r -a ALL_PAIRS <<< "$ALL_PAIRS_STR"
read -r -a PENDING_PAIRS <<< "$PENDING_STR"

echo "Run:          $RUN_TAG"
echo "Host:         $(hostname)"
echo "Started:      $(date)"
echo "Mode:         $MODE"
echo "Presets:      ${#MODEL_LIST[@]}: ${MODEL_LIST[*]}"
echo "Pairs:        ${#ALL_PAIRS[@]} selected, ${#PENDING_PAIRS[@]} with work left"
echo "Pending:      ${PENDING_PAIRS[*]:-(none)}"
echo "Overwrite:    $OVERWRITE"
echo "Workers/GPU:  $WORKERS_PER_GPU ($NSLOTS training processes on $NGPU GPU(s))"
echo "Dataset:      $DATASET_ROOT (${#DATA_FILES[@]} files)"
echo "Results:      $PROJECT_DIR/results/$MODE"
nvidia-smi -L

# ---- Step 1: preflight -----------------------------------------------------------
PREFLIGHT_ARGS=(--dataset-root "$DATASET_ROOT" --mode "$MODE" --models "${MODEL_LIST[@]}"
                --n-slots "$NSLOTS" --walltime-hours "$FULL_WALLTIME_H")
if [ "$MODE" != "smoke" ]; then
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

# ---- Step 2: training, pairs dealt round-robin over the worker slots --------------------
echo ""
echo "Step 2: training $(date)"
if [ ${#PENDING_PAIRS[@]} -gt 0 ]; then
    GROUP_PAIRS=()
    for i in "${!PENDING_PAIRS[@]}"; do
        s=$((i % NSLOTS))
        GROUP_PAIRS[s]="${GROUP_PAIRS[s]:-} ${PENDING_PAIRS[i]}"
    done
    PIDS=()
    for s in "${!GROUP_PAIRS[@]}"; do
        g=$((s % NGPU))
        w=$((s / NGPU))
        LOG="${LOGDIR}/gpu${g}_w${w}_${RUN_TAG}.log"
        echo "GPU $g worker $w: pairs=[${GROUP_PAIRS[s]# }] -> $LOG"
        # word splitting of the pair group is intentional
        CUDA_VISIBLE_DEVICES=$g python -m scripts.run_selected_pairs \
            --pairs ${GROUP_PAIRS[s]} \
            --models "${MODEL_LIST[@]}" \
            --mode "$MODE" \
            --dataset-root "$DATASET_ROOT" \
            --no-log-file ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
            > "$LOG" 2>&1 &
        PIDS+=($!)
    done
    echo "Workers launched: PIDs ${PIDS[*]}  (follow one with: tail -f ${LOGDIR}/gpu0_w0_${RUN_TAG}.log)"
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAIL=1
    done
    echo "Workers finished: $(date)"
    if [ $FAIL -ne 0 ]; then
        echo "A worker failed. Its traceback is in:" >&2
        grep -l "FAILED\|Traceback" "${LOGDIR}"/gpu*_"${RUN_TAG}".log >&2 || true
    fi
else
    echo "Nothing to train: every selected (pair, preset) run is complete."
fi

# ---- Step 3: output check ---------------------------------------------------------------
echo ""
echo "Step 3: checking outputs $(date)"
python -m scripts.check_results --mode "$MODE" --pairs "${ALL_PAIRS[@]}" \
    --models "${MODEL_LIST[@]}" --dataset-root "$DATASET_ROOT" || FAIL=1

# ---- Step 4: summary (every preset with results in this mode) ---------------------------
echo ""
echo "Step 4: summary $(date)"
python -m scripts.summarize --mode "$MODE" --pairs "${ALL_PAIRS[@]}" || FAIL=1

if [ "$MODE" = "smoke" ]; then
    LEFT=$(python -m scripts.run_selected_pairs --mode smoke --pairs "${ALL_PAIRS[@]}" \
        --models "${MODEL_LIST[@]}" --list-pairs --pending)
    if [ -n "$LEFT" ]; then
        echo "Resume check FAILED: still listed as pending after training: $LEFT" >&2
        FAIL=1
    else
        echo "Resume check passed: finished runs are recognized and would be skipped."
    fi
fi

echo ""
if [ $FAIL -ne 0 ]; then
    echo "One or more steps failed - see ${LOGDIR}/*_${RUN_TAG}.log" >&2
    [ "$MODE" = "smoke" ] && echo "SMOKE TEST FAILED - do not start the screen yet." >&2
    exit 1
fi
if [ "$MODE" = "smoke" ]; then
    echo "SMOKE TEST PASSED $(date)"
    echo "Time projection: see 'timing projection' in $LOGDIR/preflight_${RUN_TAG}.log"
    echo "Next: WORKERS_PER_GPU=3 bash run_vast.sh screen"
else
    echo "All runs completed $(date): results/$MODE/summary/summary.md"
fi
