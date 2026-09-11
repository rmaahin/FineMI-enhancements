#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Riemannian (Technique 3) sweep driver - TACC GPU node or a local GPU box.
#
# Everything it produces lands in ONE folder inside Riemannian/:  ./results
#
#   ./run_riemannian.sh check     preflight only: env, GPUs, dataset, imports
#   ./run_riemannian.sh selftest  check + run_local.py --selftest (no training)
#   ./run_riemannian.sh smoke     check + selftest + 2 pairs x all arms x 2 subjects
#                                 x 5 epochs (nothing saved; proves the loop runs)
#   ./run_riemannian.sh probe     check + selftest + ONE real pair per arm (saved and
#                                 reused), prints the projected sweep time
#   ./run_riemannian.sh full      the whole thing (default): raw baseline (reused from
#                                 the FBCNet sweep when available) + every Riemannian
#                                 arm x 28 pairs, sharded across GPUs, then analysis
#   ./run_riemannian.sh analyze   just rebuild the tables from ./results
#
# Arms (see fine_mi.py):
#   ts tsimg fbts fbtsimg   FINE network on Riemannian features   (GPU)
#   mdrm tslda              classical Riemannian references       (CPU, minutes)
#   raw                     the unchanged FINE baseline
#
# Env overrides:
#   PY=/path/to/python              interpreter (default: ../morlet-scalogram/.venv if it
#                                   exists, else python)
#   FINE_DATASET_ROOT=/path/FineMI  dir with subject*_eeg_epochs_*.npz
#                                   (default: ../FineMI or ../../FineMI)
#   RESULTS_DIR=/path               results folder (default: ./results)
#   ARMS="ts tsimg"                 Riemannian arms to run (default: all six)
#   CHANNEL_SETS="all"              channel sets for those arms (default: "all motor";
#                                   'motor' results go to ${RESULTS_DIR}_motor)
#   REUSE_RAW=0                     do NOT copy ../FBCNet/results/raw; recompute raw
#   SHARDS=6                        parallel workers (default: one per GPU)
#   EXTRA="--max-subjects 4"        passed through to run_local.py
#   FINE_RIEM_CHANNELS=motor ...    any fine_mi.py knob (use a separate RESULTS_DIR!)
#
# Safe to interrupt: completed pairs are skipped on restart, so Ctrl+C and
# re-running the same command loses at most the pairs in flight.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MODE="${1:-full}"
EXTRA="${EXTRA:-}"
ARMS="${ARMS:-mdrm tslda ts tsimg fbts fbtsimg}"   # cheap arms first
REUSE_RAW="${REUSE_RAW:-1}"
# Channel sets for the Riemannian arms, both pre-specified from the papers:
#   all   - all 62 channels (the same input FINE gets)
#   motor - Tensor-CSPNet's 20 sensorimotor channels (Congedo 2017 sec. 4: N >= 32
#           electrodes swamps the Riemannian distance with irrelevant components)
CHANNEL_SETS="${CHANNEL_SETS:-all motor}"
[[ -n "${FINE_RIEM_CHANNELS:-}" ]] && CHANNEL_SETS="$FINE_RIEM_CHANNELS"

if [[ -z "${PY:-}" ]]; then
    if   [[ -x "$HERE/../morlet-scalogram/.venv/bin/python" ]];         then PY="$HERE/../morlet-scalogram/.venv/bin/python"
    elif [[ -x "$HERE/../morlet-scalogram/.venv/Scripts/python.exe" ]]; then PY="$HERE/../morlet-scalogram/.venv/Scripts/python.exe"
    else PY=python; fi
fi

if [[ -z "${FINE_DATASET_ROOT:-}" ]]; then
    FINE_DATASET_ROOT="$HERE/../FineMI"
    for cand in "$HERE/../FineMI" "$HERE/../../FineMI"; do
        if compgen -G "$cand/subject*_eeg_epochs_*.npz" > /dev/null; then
            FINE_DATASET_ROOT="$(cd "$cand" && pwd)"; break
        fi
    done
fi
export FINE_DATASET_ROOT
DATASET_ROOT="$FINE_DATASET_ROOT"
RESULTS_DIR="${RESULTS_DIR:-$HERE/results}"
LOGDIR="$RESULTS_DIR/logs"
mkdir -p "$LOGDIR"

NGPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
[[ "$NGPU" -lt 1 ]] && NGPU=1
SHARDS="${SHARDS:-$NGPU}"

# The geometry pins BLAS to one thread per call; give each shard a fair share of cores
# for everything else (torch CPU ops, FFTs).
NCPU="$(nproc 2>/dev/null || echo 4)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( NCPU / SHARDS > 0 ? NCPU / SHARDS : 1 ))}"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOGDIR/sweep.log"; }
hr()  { printf '=%.0s' $(seq 1 72); echo; }

trap 'say "INTERRUPTED - rerun the same command to resume"; exit 130' INT TERM

hr
say "Riemannian sweep  |  mode=$MODE"
say "  host    : $(hostname)"
say "  python  : $("$PY" -c 'import sys; print(sys.executable)')"
say "  repo    : $HERE"
say "  dataset : $DATASET_ROOT"
say "  results : $RESULTS_DIR"
say "  arms    : raw(+reuse=$REUSE_RAW) $ARMS  x channels {$CHANNEL_SETS}"
say "  gpus    : $NGPU   shards: $SHARDS   OMP_NUM_THREADS: $OMP_NUM_THREADS"
hr

# ------------------------------------------------------------------ preflight
say "PREFLIGHT"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOGDIR/preflight.log"
import os, sys
import fine_mi as F                      # MUST be imported before torch
import torch, sklearn, scipy, numpy, threadpoolctl

root = os.environ['FINE_DATASET_ROOT']
print(f"python      : {sys.version.split()[0]}")
print(f"torch       : {torch.__version__}  cuda={torch.version.cuda}")
print(f"numpy {numpy.__version__} | scipy {scipy.__version__} | sklearn {sklearn.__version__}"
      f" | threadpoolctl {threadpoolctl.__version__}")
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
names = F.channel_names(root)
print(f"montage: {len(names)} entries in channel_location_64_neuroscan.locs")
print(f"pipeline_version={F.PIPELINE_VERSION} raw_pipeline_version={F.RAW_PIPELINE_VERSION} "
      f"cue_sample={F.CUE_SAMPLE}")
PYEOF
say "preflight ok"

[[ "$MODE" == "check" ]] && { say "check only - stopping"; exit 0; }

analyze () {
    "$PY" analyze_results.py --results-root "$RESULTS_DIR" 2>&1 | tee "$LOGDIR/analyze.log"
    say "tables written to $RESULTS_DIR/analysis/"
}

if [[ "$MODE" == "analyze" || "$MODE" == "compare" ]]; then
    analyze
    exit 0
fi

# ------------------------------------------------------------------ self-test
say "SELF-TEST (SPD geometry, Ledoit-Wolf, rank handling, filter bank, every arm fwd/bwd)"
"$PY" run_local.py --selftest --dataset-root "$DATASET_ROOT" \
    --results-root "$RESULTS_DIR" 2>&1 | tee "$LOGDIR/selftest.log"
grep -q "ALL SELF-TESTS PASSED" "$LOGDIR/selftest.log" || { say "SELF-TEST FAILED"; exit 1; }
say "self-test passed"

[[ "$MODE" == "selftest" ]] && { say "selftest only - stopping"; exit 0; }

ARMS_CSV="$(echo "$ARMS" | tr -s ' ' ',' | sed 's/^,//; s/,$//')"

# ---------------------------------------------------------------------- smoke
if [[ "$MODE" == "smoke" ]]; then
    say "SMOKE (2 pairs, raw + $ARMS_CSV, 2 subjects, 5 epochs - not saved)"
    "$PY" run_local.py --mode "raw,$ARMS_CSV" --pairs HOC/SPS,SAA/SFE --smoke \
        --dataset-root "$DATASET_ROOT" --results-root "$RESULTS_DIR" $EXTRA \
        2>&1 | tee "$LOGDIR/smoke.log"
    say "smoke passed - the training loop runs end to end on this node"
    exit 0
fi

# ------------------------------------------------------------ raw baseline reuse
# The raw arm here is the FBCNet/Morlet raw arm, bit-for-bit (verified: identical input
# tensors, same folds/seeds). Copying their saved raw results skips ~1.2 GPU-hours.
# They are only bit-identical to a fresh run on the same GPU model; the analysis script
# reports whether the pairing is exact.
if [[ "$REUSE_RAW" == "1" ]]; then
    for src in "$HERE/../FBCNet/results/raw" "$HERE/../morlet-scalogram/results/raw"; do
        if [[ -d "$src" ]]; then
            mkdir -p "$RESULTS_DIR/raw"
            n_new=0
            for d in "$src"/*/; do
                slug="$(basename "$d")"
                if [[ -f "$d/predictions.csv" && ! -d "$RESULTS_DIR/raw/$slug" ]]; then
                    cp -r "$d" "$RESULTS_DIR/raw/$slug"
                    n_new=$(( n_new + 1 ))
                fi
            done
            gpu_src=$( (grep -h '"gpu"' "$src"/*/results.json 2>/dev/null || true) \
                       | sort -u | tr -d ' ,' | head -1)
            say "raw baseline: reused $n_new new pair(s) from $src (${gpu_src:-gpu unknown})"
            break
        fi
    done
fi

# Each channel set is its own results folder: 'all' -> $RESULTS_DIR, anything else ->
# ${RESULTS_DIR}_<set> (e.g. results_motor). analyze_results.py picks the siblings up as
# extra arms named e.g. ts@motor, so everything lands in one paired table.
set_dir () { if [[ "$1" == "all" ]]; then echo "$RESULTS_DIR"; else echo "${RESULTS_DIR}_$1"; fi; }

# ---------------------------------------------------------------------- probe
if [[ "$MODE" == "probe" || "$MODE" == "full" ]]; then
    say "TIMING PROBE (HOC/SPS, one pair per arm; results ARE saved and reused)"
    for cs in $CHANNEL_SETS; do
        modes_cs="$ARMS_CSV"; [[ "$cs" == "all" ]] && modes_cs="raw,$ARMS_CSV"
        FINE_RIEM_CHANNELS="$cs" "$PY" run_local.py --mode "$modes_cs" --pairs HOC/SPS \
            --time-probe --dataset-root "$DATASET_ROOT" --results-root "$(set_dir "$cs")" \
            $EXTRA 2>&1 | tee "$LOGDIR/probe_${cs}.log"
    done
    say "probe done - projected sweep times are in $LOGDIR/probe_*.log"
    [[ "$MODE" == "probe" ]] && { say "probe only - stopping"; exit 0; }
fi

[[ "$MODE" == "full" ]] || { say "unknown mode '$MODE'"; exit 2; }

# ------------------------------------------------------------------ the sweep
# 28 pairs as index tokens ("0-1" ... "6-7"), dealt round-robin to the shards.
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

run_arm () {                      # $1 = arm name, $2 = channel set
    local arm="$1" cs="${2:-all}" t0 t1 rc=0 dir tag
    dir="$(set_dir "$cs")"; tag="${arm}_${cs}"
    t0=$(date +%s)
    say "ARM '$arm' [channels=$cs] starting: 28 pairs over $SHARDS shard(s) -> $dir"
    if (( SHARDS <= 1 )); then
        FINE_RIEM_CHANNELS="$cs" "$PY" run_local.py --mode "$arm" --pairs all \
            --dataset-root "$DATASET_ROOT" --results-root "$dir" $EXTRA \
            2>&1 | tee "$LOGDIR/${tag}.log" || rc=$?
    else
        local pids=() k gpu plist n
        for k in $(seq 1 "$SHARDS"); do
            gpu=$(( (k - 1) % NGPU ))
            plist="$(shard_pairs "$k")"
            [[ -z "$plist" ]] && continue
            CUDA_VISIBLE_DEVICES="$gpu" FINE_RIEM_CHANNELS="$cs" \
                "$PY" run_local.py --mode "$arm" --pairs "$plist" \
                    --dataset-root "$DATASET_ROOT" --results-root "$dir" $EXTRA \
                    > "$LOGDIR/${tag}_shard${k}.log" 2>&1 &
            pids+=($!)
            n=$(awk -F, '{print NF}' <<< "$plist")
            say "  shard $k/$SHARDS -> gpu $gpu | pid ${pids[-1]} | $n pairs | $LOGDIR/${tag}_shard${k}.log"
        done
        for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
    fi
    (( rc != 0 )) && say "  WARNING: arm '$arm' [$cs] exited non-zero (rc=$rc) - check its log"
    t1=$(date +%s)
    say "ARM '$arm' [channels=$cs] finished in $(( (t1 - t0) / 60 )) min"
}

run_arm raw all                   # no-op for every pair already reused / saved
for cs in $CHANNEL_SETS; do
    for arm in $ARMS; do run_arm "$arm" "$cs"; done
done

# -------------------------------------------------------------------- results
say "ANALYZING"
analyze

hr
say "SWEEP COMPLETE"
n=$(find "$RESULTS_DIR/raw" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
say "  raw: $n/28 pairs saved"
for cs in $CHANNEL_SETS; do
    for m in $ARMS; do
        n=$(find "$(set_dir "$cs")/$m" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        say "  $m [channels=$cs]: $n/28 pairs saved"
    done
done
say "tables : $RESULTS_DIR/analysis/arms_summary.csv, per_pair.csv, groups.csv, comparison.csv"
say "copy these folders off the node before the job ends: $RESULTS_DIR*"
hr
