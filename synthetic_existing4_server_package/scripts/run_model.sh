#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?usage: run_model.sh MODEL_SLUG [OUTPUT_DIR]}"
OUT="${2:-$ROOT/results}"
mkdir -p "$OUT/logs"
python "$ROOT/src/run_synthetic.py" \
  --config "$ROOT/config/experiment.json" --data-dir "$ROOT/data" \
  --output-dir "$OUT" --model "$MODEL" \
  --budgets 0,64,128,256,384,512,768 --option-order-seeds 11,23,37 \
  2>&1 | tee -a "$OUT/logs/${MODEL}.log"
