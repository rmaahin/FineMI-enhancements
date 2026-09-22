#!/bin/bash
#SBATCH -J gru_fine
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o results/slurm_%j.out
#SBATCH -e results/slurm_%j.err
# Allocation: add `-A <project>` to the sbatch command if you have more than one.
#
# FINE + GRU and FINE + GRU + CWT (Conformer B / B + CWT with the transformer
# replaced by a 2-layer BiGRU) on FineMI 0.5-3 Hz, decimated 250 -> 50 Hz (D = 5),
# on a Lonestar6 A100 node (3 GPUs; no --gres, the job gets
# the whole node).
#
# Submit from GRU_FINE/ (results and logs are written relative to it):
#
#   cd $WORK/FineMI-enhancements/GRU_FINE
#
#   # 1. smoke test on the dev queue (~15-30 min)
#   MODE=smoke sbatch -p gpu-a100-dev -t 01:00:00 submit_ls6.slurm
#
#   # 2. full sweep on gpu-a100, once the smoke job ends with "SMOKE TEST PASSED"
#   sbatch submit_ls6.slurm
#
# What each mode does:
#   smoke  1. preflight: packages, GPU, determinism, ALL 18 subject files, CWT checks,
#             every model x every window at full-sweep shapes, and a timed projection of
#             the full sweep vs FULL_WALLTIME_H
#          2. trains 3 pairs (one per GPU) x 2 models through the SAME code path as the
#             full sweep (5-fold -> 2-fold, 50 -> 2 epochs, 2 subjects, all 4 windows)
#          3. comparison table + paired stats + comparison with Conformer A / B / B+CWT
#             (../Conformer_decimated_CWT/results), output check, resume check
#          -> results/smoke/
#   full   1. preflight without the timing projection (fail fast before 24 h of work)
#          2. all 28 pairs x 2 models, pairs dealt round-robin over the 3 GPUs
#          3. comparison table + paired stats + comparison with Conformer A / B / B+CWT
#             + output check
#          -> results/full/
#
# Settings (environment variables, read at submit time; sbatch exports your env):
#   MODE             full (default) | smoke
#   PAIRS            full: "all" (default); smoke: "HOC_WFE WAA_SAA EPS_SPS" (default).
#                    Space-separated, e.g. WAA_SAA HOC_WFE (or 2_6, or index 0-27)
#   MODELS           default "fine_gru fine_gru_cwt"
#   DATASET_ROOT     default $SCRATCH/finemi-dataset/FineMI_0.5_3hz
#   VENV             default $SCRATCH/venvs/fine
#   OVERWRITE        1 = re-train runs that already have results (default: 1 in smoke, 0 in full)
#   FULL_WALLTIME_H  hours the full job may run; the smoke projection must fit in it.
#                    Default 12, keep equal to the "#SBATCH -t" line above.
#
# Resume: (pair, model) runs whose results.json carries this pipeline's stamp are
# skipped, and pairs already complete for every model are dropped before the work is
# split across GPUs. If the full job times out, resubmit the same command.

set -euo pipefail

# ---- Settings ------------------------------------------------------------
MODE="${MODE:-full}"
case "$MODE" in
  smoke)
    PAIRS="${PAIRS:-HOC_WFE WAA_SAA EPS_SPS}"
    OVERWRITE="${OVERWRITE:-1}" ;;
  full)
    PAIRS="${PAIRS:-all}"
    OVERWRITE="${OVERWRITE:-0}" ;;
  *) echo "Unknown MODE=$MODE (expected smoke|full)" >&2; exit 1 ;;
esac
MODELS="${MODELS:-fine_gru fine_gru_cwt}"
DATASET_ROOT="${DATASET_ROOT:-$SCRATCH/finemi-dataset/FineMI_0.5_3hz}"
VENV="${VENV:-$SCRATCH/venvs/fine}"
FULL_WALLTIME_H="${FULL_WALLTIME_H:-12}"

# ---- Paths ---------------------------------------------------------------
REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"
if [ ! -f scripts/run_selected_pairs.py ] || [ ! -f src/models/fine_gru_cwt.py ]; then
    echo "Run sbatch from the GRU_FINE/ directory (got $REPO_ROOT)" >&2
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
# `python -m scripts.*` from REPO_ROOT already makes `import src.*` resolve;
# PYTHONPATH is set too so it also works if a worker is started another way.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

EXTRA_ARGS=()
if [ "$OVERWRITE" = "1" ]; then
    EXTRA_ARGS+=(--overwrite)
fi

read -r -a MODEL_LIST <<< "$MODELS"

NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -lt 1 ]; then
    echo "No GPUs visible on $(hostname)" >&2
    exit 1
fi

# Validate + normalize pairs ("all" -> 28, SAA_WAA -> WAA_SAA, duplicates dropped).
# ALL_PAIRS is everything selected (for the tables); PENDING_PAIRS drops pairs that
# are already complete for every model, so GPUs aren't handed finished work.
ALL_PAIRS_STR=$(python -m scripts.run_selected_pairs --pairs $PAIRS --list-pairs \
    --models "${MODEL_LIST[@]}" --mode "$MODE")
PENDING_STR=$(python -m scripts.run_selected_pairs --pairs $PAIRS --list-pairs --pending \
    --models "${MODEL_LIST[@]}" --mode "$MODE" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
read -r -a ALL_PAIRS <<< "$ALL_PAIRS_STR"
read -r -a PENDING_PAIRS <<< "$PENDING_STR"

echo "Job:          $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Partition:    ${SLURM_JOB_PARTITION:-?}"
echo "Started:      $(date)"
echo "Mode:         $MODE"
echo "Models:       ${MODEL_LIST[*]}"
echo "Pairs:        ${#ALL_PAIRS[@]} selected, ${#PENDING_PAIRS[@]} with runs still to do"
echo "Pending:      ${PENDING_PAIRS[*]:-(none)}"
echo "Overwrite:    $OVERWRITE"
echo "Dataset:      $DATASET_ROOT (${#DATA_FILES[@]} files)"
echo "Decimation:   D=5 (250 -> 50 Hz)"
echo "Results:      $REPO_ROOT/results/$MODE"
echo "Venv:         $VENV"
echo "GPUs:"
nvidia-smi -L

# ---- Step 1: preflight (stops the job before any training if it fails) -------
PREFLIGHT_ARGS=(--dataset-root "$DATASET_ROOT" --mode "$MODE"
                --models "${MODEL_LIST[@]}" --n-gpus "$NGPU"
                --walltime-hours "$FULL_WALLTIME_H")
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

# ---- Step 2: training, one worker per GPU ------------------------------------
echo ""
echo "Step 2: training $(date)"
if [ ${#PENDING_PAIRS[@]} -gt 0 ]; then
    # 1 pending pair  -> one architecture per GPU, in parallel
    # 2+ pending pairs -> pairs dealt round-robin across the GPUs; each worker trains
    #                     every model for its pairs in turn
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
    fi

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
    if [ $FAIL -ne 0 ]; then
        echo "A worker failed. Its traceback is in ${LOGDIR}/gpu*_${SLURM_JOB_ID}.log:" >&2
        grep -l "FAILED\|Traceback" "${LOGDIR}"/gpu*_"${SLURM_JOB_ID}".log >&2 || true
    fi
else
    echo "Nothing to train: every selected pair is complete for every model."
fi

# ---- Step 3: comparison table + paired tests ----------------------------------
echo ""
echo "Step 3: comparison table $(date)"
python -m scripts.make_comparison_table \
    --pairs "${ALL_PAIRS[@]}" --models "${MODEL_LIST[@]}" --mode "$MODE" || FAIL=1

if [ ${#MODEL_LIST[@]} -gt 1 ]; then
    echo ""
    echo "Computing paired statistics..."
    python -m scripts.paired_stats \
        --pairs "${ALL_PAIRS[@]}" --models "${MODEL_LIST[@]}" --mode "$MODE" || FAIL=1
fi

# Needs the finished Conformer_decimated_CWT sweep (results/<mode>/) next to this
# folder; a missing reference only costs this report, so it does not fail the job.
echo ""
echo "Comparing with Conformer A / B / B + CWT..."
python -m scripts.compare_to_reference     --pairs "${ALL_PAIRS[@]}" --models "${MODEL_LIST[@]}" --mode "$MODE"     || echo "WARNING: compare_to_reference failed (reference results missing?); see above" >&2

# ---- Step 4: output check (+ resume check in smoke) ---------------------------
echo ""
echo "Step 4: checking outputs $(date)"
python -m scripts.check_results --mode "$MODE" --pairs "${ALL_PAIRS[@]}" \
    --models "${MODEL_LIST[@]}" --dataset-root "$DATASET_ROOT" || FAIL=1

if [ "$MODE" = "smoke" ]; then
    # A resubmitted full job relies on finished runs being recognized. After a
    # complete smoke run, nothing may be listed as pending.
    LEFT=$(python -m scripts.run_selected_pairs --pairs "${ALL_PAIRS[@]}" --list-pairs \
        --pending --models "${MODEL_LIST[@]}" --mode smoke)
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
    echo "Full-sweep projection: see the 'timing projection' line in $LOGDIR/preflight_${SLURM_JOB_ID}.log"
    echo "Next: sbatch submit_ls6.slurm   (from $REPO_ROOT, with MODE and PAIRS unset)"
else
    echo "All runs completed $(date): results/full/comparison/comparison_table.md, paired_stats.md and vs_reference.md"
fi
