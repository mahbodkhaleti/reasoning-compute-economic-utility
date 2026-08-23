#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${QWEN14_PYTHON_BIN:-${PACKAGE_DIR}/.venv/bin/python}"
SOURCE_DATA_DIR="${QWEN14_SOURCE_DATA_DIR:-${PACKAGE_DIR}/../robustness_extensions/server_outputs/data}"
SMOKE_DIR="${QWEN14_SMOKE_DIR:-${PACKAGE_DIR}/smoke_results}"

mkdir -p "${SMOKE_DIR}/data"
for name in scenarios.csv auction_splits.csv dataset_summary.csv metadata.json; do
  if [[ -f "${SOURCE_DATA_DIR}/${name}" ]]; then
    cp "${SOURCE_DATA_DIR}/${name}" "${SMOKE_DIR}/data/${name}"
  fi
done

"${PYTHON_BIN}" "${PACKAGE_DIR}/scripts/check_environment.py"
"${PYTHON_BIN}" "${PACKAGE_DIR}/scripts/validate_qwen3_14b.py" --output-dir "${SMOKE_DIR}" --stage inputs
"${PYTHON_BIN}" "${PACKAGE_DIR}/scripts/run_fixed_budget.py" \
  --config "${PACKAGE_DIR}/config.json" --output-dir "${SMOKE_DIR}" --model qwen3_14b \
  --chat-mode reasoning --decoding greedy --budgets 0,128 --repeat-seeds 11 \
  --limit 2 --constrained-option-readout

echo "Smoke test passed. Its outputs are isolated under ${SMOKE_DIR} and are not paper data."
