#!/bin/bash
# Batch extract Mathlib declarations using jixia.
# Usage: cd mathlib4 && bash ../src/scripts/jixia_batch.sh [JOBS]
set -euo pipefail

JIXIA="$(cd "$(dirname "$0")/../../jixia/.lake/build/bin" && pwd)/jixia"
OUTDIR="$(cd "$(dirname "$0")/../../data/jixia_decls" && pwd)"
JOBS="${1:-8}"

echo "jixia binary: $JIXIA"
echo "Output dir:   $OUTDIR"
echo "Parallel jobs: $JOBS"

mkdir -p "$OUTDIR"

# Generate file list (only files that haven't been processed yet)
find Mathlib -name "*.lean" | while IFS= read -r f; do
    outname="$(echo "$f" | sed 's|/|_|g; s|\.lean$|.json|')"
    if [ ! -f "$OUTDIR/$outname" ]; then
        echo "$f"
    fi
done > /tmp/jixia_todo.txt

total=$(wc -l < /tmp/jixia_todo.txt)
echo "Files to process: $total"

if [ "$total" -eq 0 ]; then
    echo "All files already processed."
    exit 0
fi

# Process files in parallel using background jobs with a semaphore
count=0
running=0
while IFS= read -r f; do
    outname="$(echo "$f" | tr '/' '_' | sed 's/\.lean$/.json/')"
    (lake env "$JIXIA" -d "$OUTDIR/$outname" -i "$f" 2>/dev/null || true) &
    running=$((running + 1))
    count=$((count + 1))
    if [ "$running" -ge "$JOBS" ]; then
        wait -n 2>/dev/null || wait
        running=$((running - 1))
    fi
    if [ $((count % 100)) -eq 0 ]; then
        echo "  Launched $count / $total ..."
    fi
done < /tmp/jixia_todo.txt
wait

processed=$(find "$OUTDIR" -name "*.json" | wc -l)
echo "Done. Processed files: $processed"
