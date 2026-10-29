#!/bin/bash
# LEAN + EEG Conformer on OFNER, on one Vast.ai instance with one or more GPUs. Plain
# bash, no SLURM. Run it from anywhere; it changes into this script's folder. Run it
# inside tmux so it survives an SSH disconnect (see README.md).
#
#   bash run_vast.sh smoke          # both bands: preflight + timing projection, 3 pairs
#                                   # x 2 models (2 subjects, 2 folds, 2 epochs), checks
#   bash run_vast.sh full           # both bands: all 15 pairs x 2 models + numbers
#   bash run_vast.sh full 0.5_3hz   # one band only (0.5_3hz or 8_30hz)
#
# Settings (environment variables; see run_pipeline.sh for the rest):
#   DATA_ROOT        default /workspace/ofner (data in $DATA_ROOT/ofner_mi_<band>)
#   VENV             default /workspace/venv (skipped if absent)
#   PAIRS, MODELS, OVERWRITE, FULL_WALLTIME_H   passed through to run_pipeline.sh
#
# Resume: finished (pair, model) runs are skipped, so after an interruption just run
# the same command again.

set -euo pipefail

MODE="${1:-}"
BANDS="${2:-both}"
case "$MODE" in
  smoke|full) ;;
  *) echo "Usage: bash run_vast.sh smoke|full [0.5_3hz|8_30hz|both]" >&2; exit 1 ;;
esac
[ "$BANDS" = "both" ] && BANDS="0.5_3hz 8_30hz"

DATA_ROOT="${DATA_ROOT:-/workspace/ofner}"
export VENV="${VENV:-/workspace/venv}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FAIL=0
for band in $BANDS; do
    band="${band//-/_}"; band="${band%hz}hz"
    echo ""
    echo "=================== $MODE: $band ==================="
    # unset RUN_TAG so each band gets its own log tag
    env -u RUN_TAG DATASET_ROOT="$DATA_ROOT/ofner_mi_$band" \
        bash "$PROJECT_DIR/run_pipeline.sh" "$MODE" "$band" || FAIL=1
done

echo ""
if [ $FAIL -ne 0 ]; then
    echo "At least one band failed - see results/<band>/$MODE/logs/" >&2
    exit 1
fi
if [ "$MODE" = "smoke" ]; then
    echo "SMOKE TEST PASSED for: $BANDS"
    echo "Next: bash run_vast.sh full"
else
    echo "Done: results/<band>/full/comparison/ofner_numbers.md for: $BANDS"
fi
