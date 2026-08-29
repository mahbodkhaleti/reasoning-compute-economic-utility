#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-qwen3_14b}"
OUT="$ROOT/smoke_results/$MODEL"
mkdir -p "$OUT/historical/data" "$OUT/synthetic"
cp "$ROOT/data/historical/scenarios.csv" "$OUT/historical/data/scenarios.csv"
cp "$ROOT/data/historical/auction_splits.csv" "$OUT/historical/data/auction_splits.csv"
CMD=(python "$ROOT/src/run_fixed_budget.py" --config "$ROOT/config/experiment.json" --output-dir "$OUT/historical" --model "$MODEL" --chat-mode reasoning --decoding greedy --budgets 64 --repeat-seeds 11 --limit 1 --constrained-option-readout)
"${CMD[@]}"
BEFORE="$(wc -l < "$OUT/historical/raw/fixed_${MODEL}_reasoning_greedy.jsonl")"
"${CMD[@]}"
AFTER="$(wc -l < "$OUT/historical/raw/fixed_${MODEL}_reasoning_greedy.jsonl")"
test "$BEFORE" = 1 && test "$AFTER" = "$BEFORE"
python "$ROOT/src/run_synthetic.py" --config "$ROOT/config/experiment.json" --data-dir "$ROOT/data/synthetic" --output-dir "$OUT/synthetic" --model "$MODEL" --budgets 64 --option-order-seeds 11 --limit 1
python "$ROOT/src/verify_smoke.py" --root "$OUT" --model "$MODEL"
echo "Smoke test passed for $MODEL, including resume without duplication"
