#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; MODEL="${1:?model slug required}"; CFG="$ROOT/config/experiment.json"; OUT="$ROOT/results/historical"
mkdir -p "$ROOT/logs" "$OUT"; bash "$ROOT/scripts/prepare_historical_data.sh"; COMMON=(--config "$CFG" --output-dir "$OUT" --model "$MODEL" --constrained-option-readout)
python "$ROOT/src/run_fixed_budget.py" "${COMMON[@]}" --chat-mode reasoning --decoding greedy --budgets 0,64,128,256,512,768,unconstrained --repeat-seeds 11,23,37 2>&1 | tee -a "$ROOT/logs/${MODEL}_historical_deterministic.log"
python "$ROOT/src/run_fixed_budget.py" "${COMMON[@]}" --chat-mode no_think --decoding greedy --budgets 0 --repeat-seeds 11,23,37 2>&1 | tee -a "$ROOT/logs/${MODEL}_historical_no_think.log"
python "$ROOT/src/run_fixed_budget.py" "${COMMON[@]}" --chat-mode reasoning --decoding stochastic --budgets 128,384,768 --repeat-seeds 1,7,21,42,99 --option-order-seed 314159 --query-verbal-confidence 2>&1 | tee -a "$ROOT/logs/${MODEL}_historical_stochastic.log"
