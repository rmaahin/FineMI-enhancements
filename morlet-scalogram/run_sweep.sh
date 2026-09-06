#!/usr/bin/env bash
# Full 28-pair x 2-arm sweep. Designed to be started inside tmux and left alone.
#
#   ./run_sweep.sh                 # preflight + probe + full sweep + aggregate
#   SHARDS=4 ./run_sweep.sh        # 4 parallel workers (recommended on A100)
#   ./run_sweep.sh probe           # preflight + one timing pair, then stop
#   ./run_sweep.sh check           # preflight only
#   ./run_sweep.sh aggregate       # just rebuild the tables from saved results
#
# Env:
#   PY=/path/to/python             interpreter (default: python)
#   FINE_DATASET_ROOT=/data/FineMI directory holding subject*_eeg_epochs_*.npz
#   FINE_RESULTS_ROOT=/data/out    where results land (default: ./mi_results_v2)
#   SHARDS=4                       parallel workers (default 1)
#
# Safe to re-run: completed pairs are skipped, so Ctrl+C and restart loses at most
# the pair in flight.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-python}"
MODE="${1:-full}"

# GPUs visible to this job. On a TACC gpu-a100 node this is typically 3.
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)"
[[ "$NGPU" -lt 1 ]] && NGPU=1
# Default to one worker per GPU; override with SHARDS=N.
SHARDS="${SHARDS:-$NGPU}"

export FINE_RESULTS_ROOT="${FINE_RESULTS_ROOT:-$HERE/mi_results_v2}"
LOGDIR="$FINE_RESULTS_ROOT/logs"
mkdir -p "$LOGDIR"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOGDIR/sweep.log"; }
hr()  { printf '%.0s=' {1..70}; echo; }

trap 'say "INTERRUPTED - rerun the same command to resume"; exit 130' INT TERM

hr
say "FINE sweep starting"
say "  python  : $($PY -c 'import sys;print(sys.executable)')"
say "  results : $FINE_RESULTS_ROOT"
say "  gpus    : $NGPU"
say "  shards  : $SHARDS  ($(( (SHARDS + NGPU - 1) / NGPU )) worker(s) per GPU)"
hr

# ---------------------------------------------------------------- preflight
say "PREFLIGHT"
$PY - <<'EOF' 2>&1 | tee -a "$LOGDIR/sweep.log"
import sys, torch, fine_mi as F, os
assert torch.cuda.is_available(), "NO CUDA DEVICE - refusing to start a CPU sweep"
d, gpu = F.setup_determinism()
print(F.describe_environment(d, gpu))
print(f"dataset: {F.DATASET_ROOT}")
n = len(F.list_subject_files())
print(f"subject files: {n}")
assert n == 18, f"expected 18 subject files, found {n}"
print(f"pipeline_version={F.PIPELINE_VERSION} cue_sample={F.CUE_SAMPLE} "
      f"mtc_kernels_2d={F.MTC_KERNELS_2D} cwt_chunk={F.CWT_CHUNK}")
EOF

say "SELF-TEST"
$PY run_local.py --selftest 2>&1 | tee -a "$LOGDIR/selftest.log"
say "self-test passed"

[[ "$MODE" == "check" ]] && { say "check only - stopping"; exit 0; }

if [[ "$MODE" == "aggregate" ]]; then
    $PY aggregate.py 2>&1 | tee -a "$LOGDIR/aggregate.log"
    say "aggregate written to $FINE_RESULTS_ROOT"
    exit 0
fi

# ---------------------------------------------------------------- timing probe
if [[ "$MODE" == "probe" || "$MODE" == "full" ]]; then
    say "TIMING PROBE (one pair, both arms)"
    $PY run_local.py --mode both --pairs HOC/SPS --time-probe \
        2>&1 | tee -a "$LOGDIR/probe.log"
    say "probe done - see $LOGDIR/probe.log for the projected sweep time"
    [[ "$MODE" == "probe" ]] && { say "probe only - stopping"; exit 0; }
fi

# ---------------------------------------------------------------- the sweep
run_arm () {                      # $1 = raw|cwt
    local arm="$1" t0 t1
    t0=$(date +%s)
    say "ARM '$arm' starting ($SHARDS shard(s))"
    if [[ "$SHARDS" -le 1 ]]; then
        $PY run_local.py --mode "$arm" --pairs all 2>&1 | tee -a "$LOGDIR/$arm.log"
    else
        local pids=()
        for k in $(seq 1 "$SHARDS"); do
            gpu=$(( (k - 1) % NGPU ))     # spread shards across the node's GPUs
            CUDA_VISIBLE_DEVICES="$gpu" \
                $PY run_local.py --mode "$arm" --pairs all --shard "$k/$SHARDS" \
                > "$LOGDIR/${arm}_shard${k}.log" 2>&1 &
            pids+=($!)
            say "  shard $k/$SHARDS -> gpu $gpu, pid ${pids[-1]}, $LOGDIR/${arm}_shard${k}.log"
        done
        local rc=0
        for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
        [[ $rc -ne 0 ]] && say "  WARNING: a shard exited non-zero (rc=$rc)"
    fi
    t1=$(date +%s)
    say "ARM '$arm' finished in $(( (t1 - t0) / 60 )) min"
}

# raw first: it is the cheap arm and gives the baseline early
run_arm raw
run_arm cwt

# ---------------------------------------------------------------- results
say "AGGREGATING"
$PY aggregate.py 2>&1 | tee -a "$LOGDIR/aggregate.log"

hr
say "SWEEP COMPLETE"
say "pairs saved:"
for m in raw cwt; do
    n=$(find "$FINE_RESULTS_ROOT/$m" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    say "  $m: $n/28"
done
say "download this directory before killing the pod: $FINE_RESULTS_ROOT"
hr
