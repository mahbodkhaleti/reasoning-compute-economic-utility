#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; MODEL="${1:?model slug required}"; mkdir -p "$ROOT/logs"
python "$ROOT/src/run_synthetic.py" --config "$ROOT/config/experiment.json" --data-dir "$ROOT/data/synthetic" --output-dir "$ROOT/results/synthetic" --model "$MODEL" --budgets 0,64,128,256,384,512,768 --option-order-seeds 11,23,37 2>&1 | tee -a "$ROOT/logs/${MODEL}_synthetic.log"
