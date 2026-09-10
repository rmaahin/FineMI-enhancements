#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# FBCNet (Technique 1) sweep driver for a TACC GPU node (3x A100).
#
# Everything it produces lands in ONE folder inside FBCNet/:  ./results
#
#   ./run_tacc.sh check       preflight only: env, GPUs, dataset, imports
#   ./run_tacc.sh selftest    check + run_local.py --selftest (no training)
#   ./run_tacc.sh smoke       check + selftest + 2 pairs x 2 subjects x 5 epochs
#                             (nothing is saved; ~minutes, proves the loop runs)
#   ./run_tacc.sh probe       check + selftest + ONE real pair, both arms,
#                             prints the projected time for the full sweep
#   ./run_tacc.sh full        the whole thing: 28 pairs x {raw, fbc}, sharded
#                             across the GPUs, then compare.py
#   ./run_tacc.sh compare     just rebuild the comparison tables from ./results
#
# Env overrides:
#   PY=/path/to/python              interpreter            (default: python)
#   FINE_DATASET_ROOT=/path/FineMI  dir with subject*_eeg_epochs_*.npz
#                                   (default: ../FineMI relative to this file)
#   RESULTS_DIR=/path               results folder         (default: ./results)
#   SHARDS=6                        parallel workers       (default: one per GPU)
#   EXTRA="--max-subjects 4"        passed through to run_local.py
#
# Safe to interrupt: completed pairs are skipped on restart, so Ctrl+C and
# re-running the same command loses at most the pairs in flight.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-python}"
MODE="${1:-full}"
EXTRA="${EXTRA:-}"

DATASET_ROOT="${FINE_DATASET_ROOT:-$HERE/../FineMI}"
RESULTS_DIR="${RESULTS_DIR:-$HERE/results}"
LOGDIR="$RESULTS_DIR/logs"
mkdir -p "$LOGDIR"

NGPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
[[ "$NGPU" -lt 1 ]] && NGPU=1
SHARDS="${SHARDS:-$NGPU}"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOGDIR/sweep.log"; }
hr()  { printf '=%.0s' $(seq 1 72); echo; }

trap 'say "INTERRUPTED - rerun the same command to resume"; exit 130' INT TERM

hr
say "FBCNet sweep  |  mode=$MODE"
say "  host    : $(hostname)"
say "  python  : $($PY -c 'import sys; print(sys.executable)')"
say "  repo    : $HERE"
say "  dataset : $DATASET_ROOT"
say "  results : $RESULTS_DIR"
say "  gpus    : $NGPU"
say "  shards  : $SHARDS"
hr

# ------------------------------------------------------------------ preflight
say "PREFLIGHT"
FINE_DATASET_ROOT="$DATASET_ROOT" $PY - <<'PYEOF' 2>&1 | tee -a "$LOGDIR/preflight.log"
import os, sys
import fine_mi as F                      # MUST be imported before torch
import torch

root = os.environ['FINE_DATASET_ROOT']
print(f"python      : {sys.version.split()[0]}")
print(f"torch       : {torch.__version__}  cuda={torch.version.cuda}")
assert torch.cuda.is_available(), "NO CUDA DEVICE - refusing to start a CPU sweep"
print(f"visible gpus: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")

d, gpu = F.setup_determinism()
print(F.describe_environment(d, gpu))

files = F.list_subject_files(root)
print(f"subject files: {len(files)}  (first: {os.path.basename(files[0])})")
if len(files) != 18:
    print(f"  !! WARNING: expected 18 subjects, found {len(files)} - "
          f"results will not be comparable to the paper")
print(f"pipeline_version={F.PIPELINE_VERSION} cue_sample={F.CUE_SAMPLE} "
      f"bands={len(F.BANDS)} taps={F.FB_NUMTAPS} norm={F.FB_NORM} "
      f"chunk={F.FB_CHUNK} mtc2d={F.MTC_KERNELS_2D}")
PYEOF
say "preflight ok"

[[ "$MODE" == "check" ]] && { say "check only - stopping"; exit 0; }

if [[ "$MODE" == "compare" ]]; then
    $PY compare.py --results-root "$RESULTS_DIR" 2>&1 | tee "$LOGDIR/compare.log"
    say "comparison written to $RESULTS_DIR/comparison.csv"
    exit 0
fi

# ------------------------------------------------------------------ self-test
say "SELF-TEST (filter geometry, band selectivity, determinism, shapes, VRAM)"
$PY run_local.py --selftest --dataset-root "$DATASET_ROOT" \
    --results-root "$RESULTS_DIR" 2>&1 | tee "$LOGDIR/selftest.log"
say "self-test passed"

[[ "$MODE" == "selftest" ]] && { say "selftest only - stopping"; exit 0; }

# ---------------------------------------------------------------------- smoke
if [[ "$MODE" == "smoke" ]]; then
    say "SMOKE (2 pairs, both arms, 2 subjects, 5 epochs - not saved)"
    $PY run_local.py --mode both --pairs HOC/SPS,SAA/SFE --smoke \
        --dataset-root "$DATASET_ROOT" --results-root "$RESULTS_DIR" $EXTRA \
        2>&1 | tee "$LOGDIR/smoke.log"
    say "smoke passed - the training loop runs end to end on this node"
    exit 0
fi

# ---------------------------------------------------------------------- probe
if [[ "$MODE" == "probe" || "$MODE" == "full" ]]; then
    say "TIMING PROBE (one real pair, both arms; results ARE saved and reused)"
    $PY run_local.py --mode both --pairs HOC/SPS --time-probe \
        --dataset-root "$DATASET_ROOT" --results-root "$RESULTS_DIR" $EXTRA \
        2>&1 | tee "$LOGDIR/probe.log"
    say "probe done - projected sweep time is in $LOGDIR/probe.log"
    [[ "$MODE" == "probe" ]] && { say "probe only - stopping"; exit 0; }
fi

# ------------------------------------------------------------------ the sweep
# 28 pairs as index tokens ("0-1" ... "6-7"), dealt round-robin so each shard
# gets a mix rather than a contiguous block.
PAIRS=()
for a in $(seq 0 7); do
    for b in $(seq $((a + 1)) 7); do PAIRS+=("$a-$b"); done
done

shard_pairs () {                  # $1 = shard index (1-based) -> "0-1,2-5,..."
    local k=$1 i out=""
    for i in "${!PAIRS[@]}"; do
        if (( i % SHARDS == k - 1 )); then out+="${PAIRS[$i]},"; fi
    done
    echo "${out%,}"
}

run_arm () {                      # $1 = raw|fbc
    local arm="$1" t0 t1 rc=0
    t0=$(date +%s)
    say "ARM '$arm' starting: 28 pairs over $SHARDS shard(s)"
    if (( SHARDS <= 1 )); then
        $PY run_local.py --mode "$arm" --pairs all \
            --dataset-root "$DATASET_ROOT" --results-root "$RESULTS_DIR" $EXTRA \
            2>&1 | tee "$LOGDIR/${arm}.log"
    else
        local pids=() k gpu plist n
        for k in $(seq 1 "$SHARDS"); do
            gpu=$(( (k - 1) % NGPU ))
            plist="$(shard_pairs "$k")"
            [[ -z "$plist" ]] && continue
            CUDA_VISIBLE_DEVICES="$gpu" \
                $PY run_local.py --mode "$arm" --pairs "$plist" \
                    --dataset-root "$DATASET_ROOT" --results-root "$RESULTS_DIR" $EXTRA \
                    > "$LOGDIR/${arm}_shard${k}.log" 2>&1 &
            pids+=($!)
            n=$(awk -F, '{print NF}' <<< "$plist")
            say "  shard $k/$SHARDS -> gpu $gpu | pid ${pids[-1]} | $n pairs | $LOGDIR/${arm}_shard${k}.log"
        done
        for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
        (( rc != 0 )) && say "  WARNING: a shard exited non-zero (rc=$rc) - check its log"
    fi
    t1=$(date +%s)
    say "ARM '$arm' finished in $(( (t1 - t0) / 60 )) min"
}

run_arm raw          # cheap arm first: gives the baseline early
run_arm fbc

# -------------------------------------------------------------------- results
say "COMPARING"
$PY compare.py --results-root "$RESULTS_DIR" 2>&1 | tee "$LOGDIR/compare.log"

hr
say "SWEEP COMPLETE"
for m in raw fbc; do
    n=$(find "$RESULTS_DIR/$m" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    say "  $m: $n/28 pairs saved"
done
say "tables : $RESULTS_DIR/comparison.csv, per_subject_metrics.csv, per_fold_metrics.csv"
say "copy this folder off the node before the job ends: $RESULTS_DIR"
hr
