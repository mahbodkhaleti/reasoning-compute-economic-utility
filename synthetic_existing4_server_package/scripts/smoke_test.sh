#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-phi4_mini_reasoning}"
OUT="$ROOT/smoke_results/$MODEL"
mkdir -p "$OUT"
CMD=(python "$ROOT/src/run_synthetic.py" --config "$ROOT/config/experiment.json" --data-dir "$ROOT/data" --output-dir "$OUT" --model "$MODEL" --budgets 64 --option-order-seeds 11 --limit 2)
"${CMD[@]}"
BEFORE="$(wc -l < "$OUT/raw/synthetic_${MODEL}_deterministic.jsonl")"
"${CMD[@]}"
AFTER="$(wc -l < "$OUT/raw/synthetic_${MODEL}_deterministic.jsonl")"
test "$BEFORE" = 2 && test "$AFTER" = "$BEFORE"
python "$ROOT/src/verify_synthetic.py" --data-dir "$ROOT/data" --output-dir "$OUT" --models "$MODEL" --budgets 64 --option-order-seeds 11 --scenario-limit 2
echo "Smoke test passed, including resume without duplication: $MODEL"
