#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -ne 1 || ! -f "$1/manifest.csv" ]]; then
  echo "Usage: bash run_r1.sh /path/to/authorized/features_win" >&2
  echo "The public package does not include clinical feature tables." >&2
  exit 2
fi

INPUT_DIR="$(cd "$1" && pwd)"
OUTPUT_DIR="reproduced_r1"
mkdir -p "$OUTPUT_DIR"

python3 scripts/run_r1_nested_experiments.py \
  --feature-dir "$INPUT_DIR" --out-dir "$OUTPUT_DIR/nested" \
  --outer-folds 5 --outer-repeats 3 --inner-folds 4 \
  --top-k 16 --corr-threshold 0.85 --seed 20261012

python3 scripts/run_r1_matched_optimizer_search.py \
  --feature-dir "$INPUT_DIR" --out-dir "$OUTPUT_DIR/optimizer_search" \
  --outer-folds 5 --outer-repeats 3 --inner-folds 4 --budget 200 \
  --top-k 16 --corr-threshold 0.85 --seed 20261012

python3 scripts/run_r1_fixed_relu_baseline.py \
  --feature-dir "$INPUT_DIR" \
  --split-manifest "$OUTPUT_DIR/nested/outer_split_manifest.csv" \
  --out-dir "$OUTPUT_DIR/fixed_relu" \
  --top-k 16 --corr-threshold 0.85

python3 scripts/run_r1_fixed_relu_multiseed.py \
  --feature-dir "$INPUT_DIR" \
  --split-manifest "$OUTPUT_DIR/nested/outer_split_manifest.csv" \
  --out-dir "$OUTPUT_DIR/fixed_relu_multiseed" \
  --seeds 42 123 2026 3407 8675309 \
  --top-k 16 --corr-threshold 0.85
