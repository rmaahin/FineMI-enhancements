#!/bin/bash
# Download the OFNER upper-limb dataset (Ofner et al. 2017, PLOS One; BNCI 001-2017)
# from Zenodo record 834976. Works the same on TACC ls6 and on Vast.ai.
#
#   bash download/download_ofner.sh                # motor imagery only (150 files)
#   SESSIONS="imagination execution" bash download/download_ofner.sh   # + execution
#
# 15 subjects x 10 runs per session, one GDF file per run:
#   motorimagination_subject{1..15}_run{1..10}.gdf
#   motorexecution_subject{1..15}_run{1..10}.gdf
#
# On TACC run it on a LOGIN node (compute nodes have no outbound internet).
# Re-running is safe: files that are already complete are skipped, partial ones resume.
#
# Settings (environment variables):
#   DATA_ROOT  default $SCRATCH/ofner if $SCRATCH is set, else /workspace/ofner.
#              Files land in $DATA_ROOT/raw/.
#   SESSIONS   default "imagination"
#   SUBJECTS   default "1 2 ... 15"

set -euo pipefail

if [ -n "${SCRATCH:-}" ]; then
    DATA_ROOT="${DATA_ROOT:-$SCRATCH/ofner}"
else
    DATA_ROOT="${DATA_ROOT:-/workspace/ofner}"
fi
SESSIONS="${SESSIONS:-imagination}"
SUBJECTS="${SUBJECTS:-$(seq -s ' ' 1 15)}"
BASE_URL="https://zenodo.org/records/834976/files"
RAW_DIR="$DATA_ROOT/raw"
mkdir -p "$RAW_DIR"

echo "OFNER download"
echo "  destination: $RAW_DIR"
echo "  sessions:    $SESSIONS"
echo "  subjects:    $SUBJECTS"
echo ""

# A GDF file starts with the 3 bytes "GDF". An HTML error page or a truncated
# download does not, so this catches the common failure modes without parsing.
is_gdf() {
    [ -s "$1" ] && [ "$(head -c 3 "$1")" = "GDF" ]
}

FAILED=()
for session in $SESSIONS; do
    for s in $SUBJECTS; do
        for r in $(seq 1 10); do
            name="motor${session}_subject${s}_run${r}.gdf"
            dest="$RAW_DIR/$name"
            if is_gdf "$dest" && [ ! -f "$dest.part" ]; then
                continue
            fi
            echo "  fetching $name"
            # -C - resumes a partial .part file; the rename happens only on success
            if curl -fL --retry 5 --retry-delay 5 -C - -sS \
                    -o "$dest.part" "$BASE_URL/$name?download=1" \
               && is_gdf "$dest.part"; then
                mv "$dest.part" "$dest"
            else
                echo "  !! failed: $name" >&2
                FAILED+=("$name")
            fi
        done
    done
done

# ---- Check: every subject has 10 valid runs per session ---------------------------
echo ""
echo "Check (runs per subject; expected 10):"
BAD=0
for session in $SESSIONS; do
    line="  $session:"
    for s in $SUBJECTS; do
        n=0
        for r in $(seq 1 10); do
            is_gdf "$RAW_DIR/motor${session}_subject${s}_run${r}.gdf" && n=$((n + 1))
        done
        line="$line S${s}=$n"
        [ "$n" -eq 10 ] || BAD=1
    done
    echo "$line"
done
echo "  total size: $(du -sh "$RAW_DIR" | cut -f1)"

if [ ${#FAILED[@]} -gt 0 ] || [ $BAD -ne 0 ]; then
    echo ""
    echo "INCOMPLETE: ${#FAILED[@]} file(s) failed: ${FAILED[*]:-}" >&2
    echo "Re-run the same command to retry; finished files are skipped." >&2
    exit 1
fi
echo ""
echo "All files present. Next:"
echo "  python preprocess/preprocess_ofner.py --raw $RAW_DIR --out-root $DATA_ROOT --band both"
