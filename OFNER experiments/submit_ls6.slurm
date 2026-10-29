#!/bin/bash
#SBATCH -J ofner_lean_conformer
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o results/slurm_%j.out
#SBATCH -e results/slurm_%j.err
# Allocation: add `-A <project>` to the sbatch command if you have more than one.
#
# LEAN + EEG Conformer on OFNER on a Lonestar6 A100 node (3 GPUs; no --gres, the job
# gets the whole node). Pairs are dealt round-robin over the 3 GPUs by run_pipeline.sh.
#
# Submit from "OFNER experiments/" (results and logs are written relative to it; the
# results/ folder must exist before sbatch, for the -o/-e files above):
#
#   cd "$WORK/FineMI-enhancements/OFNER experiments"
#   mkdir -p results
#   MODE=smoke sbatch -p gpu-a100-dev -t 01:00:00 submit_ls6.slurm   # smoke, both bands
#   sbatch submit_ls6.slurm                                          # full, both bands
#   BAND=8_30hz sbatch submit_ls6.slurm                              # full, one band
#
# Settings (environment variables, read at submit time; sbatch exports your env):
#   MODE             full (default) | smoke
#   BAND             both (default) | 0.5_3hz | 8_30hz
#   DATA_ROOT        default $SCRATCH/ofner (data in $DATA_ROOT/ofner_mi_<band>)
#   VENV             default $SCRATCH/venvs/fine
#   PAIRS, MODELS, OVERWRITE   passed through to run_pipeline.sh
#   FULL_WALLTIME_H  hours the full job may run; the smoke projection (per band) must fit
#                    in it. Default 12, keep equal to the "#SBATCH -t" line above.
#
# Resume: finished pairs are skipped, so a timed-out job is simply resubmitted.

set -euo pipefail

MODE="${MODE:-full}"
BANDS="${BAND:-both}"
case "$MODE" in
  smoke|full) ;;
  *) echo "Unknown MODE=$MODE (expected smoke|full)" >&2; exit 1 ;;
esac
[ "$BANDS" = "both" ] && BANDS="0.5_3hz 8_30hz"

DATA_ROOT="${DATA_ROOT:-$SCRATCH/ofner}"
export VENV="${VENV:-$SCRATCH/venvs/fine}"
export FULL_WALLTIME_H="${FULL_WALLTIME_H:-12}"

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"
if [ ! -f run_pipeline.sh ] || [ ! -f src/models/conformer_b_slim.py ]; then
    echo "Run sbatch from the 'OFNER experiments' directory (got $REPO_ROOT)" >&2
    exit 1
fi
if [ ! -f "$VENV/bin/activate" ]; then
    echo "No virtualenv at VENV=$VENV" >&2
    exit 1
fi

echo "Job:          $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Partition:    ${SLURM_JOB_PARTITION:-?}"
echo "Mode:         $MODE"
echo "Bands:        $BANDS"

FAIL=0
for band in $BANDS; do
    band="${band//-/_}"; band="${band%hz}hz"
    echo ""
    echo "=================== $MODE: $band ==================="
    RUN_TAG="${MODE}_${band}_${SLURM_JOB_ID}" DATASET_ROOT="$DATA_ROOT/ofner_mi_$band" \
        bash run_pipeline.sh "$MODE" "$band" || FAIL=1
done

echo ""
if [ $FAIL -ne 0 ]; then
    echo "At least one band failed - see results/<band>/$MODE/logs/*_${SLURM_JOB_ID}.log" >&2
    exit 1
fi
if [ "$MODE" = "smoke" ]; then
    echo "SMOKE TEST PASSED for: $BANDS"
    echo "Next: sbatch submit_ls6.slurm   (from $REPO_ROOT, with MODE and PAIRS unset)"
else
    echo "All runs completed $(date): results/<band>/full/comparison/ofner_numbers.md"
fi
