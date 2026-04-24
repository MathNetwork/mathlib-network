#!/bin/bash
# MathlibGraph Data Extraction Pipeline
#
# Prerequisites:
#   - mathlib4 built locally (lake build completed)
#   - lean4export cloned and built
#   - lean-training-data cloned and built
#   - huggingface-cli installed and logged in (for upload)
#
# Usage:
#   ./scripts/extract.sh [OPTIONS] [MATHLIB_DIR] [LEAN4EXPORT_DIR] [TRAINING_DATA_DIR]
#
# Options:
#   --local    Output to local directory only (skip HuggingFace upload)
#   --output   Local output directory (default: ./output)

set -e

# Parse options
LOCAL_ONLY=false
LOCAL_OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            LOCAL_ONLY=true
            shift
            ;;
        --output)
            LOCAL_OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Default paths (adjust as needed)
MATHLIB_DIR="${1:-../mathlib4}"
LEAN4EXPORT_DIR="${2:-../lean4export}"
TRAINING_DATA_DIR="${3:-../lean-training-data}"

# Set output directory
if [ "$LOCAL_ONLY" = true ]; then
    OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-./output}"
    mkdir -p "$OUTPUT_DIR"
else
    OUTPUT_DIR=$(mktemp -d)
    trap "rm -rf $OUTPUT_DIR" EXIT
fi

RAW_DIR="$OUTPUT_DIR/raw"

echo "=== MathlibGraph Data Extraction Pipeline ==="
echo ""
echo "Mathlib directory: $MATHLIB_DIR"
echo "lean4export directory: $LEAN4EXPORT_DIR"
echo "lean-training-data directory: $TRAINING_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Local only: $LOCAL_ONLY"
echo ""

# Create output directories
mkdir -p "$RAW_DIR"

# Step 1: Run lean4export
echo "=== Step 1: Running lean4export ==="
ORIG_DIR=$(pwd)
cd "$MATHLIB_DIR"
if [ ! -f "$LEAN4EXPORT_DIR/.lake/build/bin/lean4export" ]; then
    echo "Building lean4export..."
    cd "$LEAN4EXPORT_DIR"
    lake build
    cd "$MATHLIB_DIR"
fi
echo "Exporting Mathlib to NDJSON (this may take a while)..."
lake env "$LEAN4EXPORT_DIR/.lake/build/bin/lean4export" Mathlib > "$ORIG_DIR/$RAW_DIR/mathlib.ndjson"
cd "$ORIG_DIR"
echo "  Created: $RAW_DIR/mathlib.ndjson"

# Step 2: Run premises
echo ""
echo "=== Step 2: Running lean-training-data premises ==="
cd "$TRAINING_DATA_DIR"
if [ ! -f ".lake/build/bin/premises" ]; then
    echo "Building lean-training-data..."
    lake exe cache get || true
    lake build
fi
echo "Extracting premises (this may take a while)..."
lake exe premises Mathlib > "$ORIG_DIR/$RAW_DIR/premises.txt"
cd "$ORIG_DIR"
echo "  Created: $RAW_DIR/premises.txt"

# Step 3: Parse lean4export → nodes.csv
echo ""
echo "=== Step 3: Parsing lean4export → nodes.csv ==="
python -m parser.from_lean4export \
    --input "$RAW_DIR/mathlib.ndjson" \
    --output "$OUTPUT_DIR/nodes.csv" \
    --filter-mathlib

# Step 4: Parse premises → edges.csv
echo ""
echo "=== Step 4: Parsing premises → edges.csv ==="
python -m parser.from_premises \
    --input "$RAW_DIR/premises.txt" \
    --output "$OUTPUT_DIR/edges.csv" \
    --filter-mathlib

# Step 5: Merge and validate
echo ""
echo "=== Step 5: Merge and validate ==="
python -m parser.merge \
    --nodes "$OUTPUT_DIR/nodes.csv" \
    --edges "$OUTPUT_DIR/edges.csv" \
    --report "$OUTPUT_DIR/summary.md"

echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.csv "$OUTPUT_DIR"/*.md 2>/dev/null || true

# Step 6: Upload to HuggingFace (unless --local)
if [ "$LOCAL_ONLY" = true ]; then
    echo ""
    echo "=== Done (local only) ==="
    echo "Files saved to: $OUTPUT_DIR"
else
    echo ""
    echo "=== Step 6: Uploading to HuggingFace ==="
    huggingface-cli upload MathNetwork/MathlibGraph "$OUTPUT_DIR/nodes.csv" nodes.csv --repo-type dataset
    huggingface-cli upload MathNetwork/MathlibGraph "$OUTPUT_DIR/edges.csv" edges.csv --repo-type dataset
    huggingface-cli upload MathNetwork/MathlibGraph "$OUTPUT_DIR/summary.md" summary.md --repo-type dataset
    echo ""
    echo "=== Done ==="
    echo "Data available at: https://huggingface.co/datasets/MathNetwork/MathlibGraph"
fi
