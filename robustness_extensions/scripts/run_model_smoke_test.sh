#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 MODEL_SLUG [RUN_ID]" >&2
  exit 2
fi
MODEL="$1"
RUN_ID="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_DIR}/robustness_extensions/smoke_outputs/${MODEL}/${RUN_ID}"
CONFIG_PATH="${SCRIPT_DIR}/config.json"
AUCTION_CSV="${PROJECT_DIR}/online-auctions-dataset/auction.csv"

if [[ -n "${ROBUSTNESS_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${ROBUSTNESS_PYTHON_BIN}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_scenarios.py" \
  --config "${CONFIG_PATH}" --auction-csv "${AUCTION_CSV}" --output-dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_fixed_budget.py" \
  --config "${CONFIG_PATH}" --output-dir "${OUTPUT_DIR}" --model "${MODEL}" \
  --chat-mode reasoning --decoding greedy --budgets 0,64,768,unconstrained \
  --repeat-seeds 11 --limit 2 --constrained-option-readout

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_fixed_budget.py" \
  --config "${CONFIG_PATH}" --output-dir "${OUTPUT_DIR}" --model "${MODEL}" \
  --chat-mode no_think --decoding greedy --budgets 0 \
  --repeat-seeds 11 --limit 2 --constrained-option-readout

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_fixed_budget.py" \
  --config "${CONFIG_PATH}" --output-dir "${OUTPUT_DIR}" --model "${MODEL}" \
  --chat-mode reasoning --decoding stochastic --budgets 128 \
  --repeat-seeds 1,7 --option-order-seed 314159 --limit 2 \
  --query-verbal-confidence --constrained-option-readout

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_smoke.py" \
  --output-dir "${OUTPUT_DIR}" --model "${MODEL}"

echo "Smoke test PASS for ${MODEL}"
echo "Smoke output: ${OUTPUT_DIR}"
