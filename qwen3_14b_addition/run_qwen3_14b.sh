#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${PACKAGE_DIR}/scripts"
CONFIG_PATH="${PACKAGE_DIR}/config.json"
RESULTS_DIR="${QWEN14_RESULTS_DIR:-${PACKAGE_DIR}/results}"
SOURCE_DATA_DIR="${QWEN14_SOURCE_DATA_DIR:-${PACKAGE_DIR}/../robustness_extensions/server_outputs/data}"
PYTHON_BIN="${QWEN14_PYTHON_BIN:-${PACKAGE_DIR}/.venv/bin/python}"
MODE="${1:-all}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found at ${PYTHON_BIN}." >&2
  echo "Run: bash ${PACKAGE_DIR}/setup_environment.sh" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}/data" "${RESULTS_DIR}/raw" "${RESULTS_DIR}/logs"
for name in scenarios.csv auction_splits.csv dataset_summary.csv metadata.json; do
  if [[ ! -f "${RESULTS_DIR}/data/${name}" && -f "${SOURCE_DATA_DIR}/${name}" ]]; then
    cp "${SOURCE_DATA_DIR}/${name}" "${RESULTS_DIR}/data/${name}"
  fi
done

if [[ ! -f "${RESULTS_DIR}/data/scenarios.csv" || ! -f "${RESULTS_DIR}/data/auction_splits.csv" ]]; then
  if [[ -n "${QWEN14_AUCTION_CSV:-}" && -f "${QWEN14_AUCTION_CSV}" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_scenarios.py" \
      --config "${CONFIG_PATH}" --auction-csv "${QWEN14_AUCTION_CSV}" --output-dir "${RESULTS_DIR}"
  else
    echo "The protocol-matched scenarios.csv and auction_splits.csv are missing." >&2
    echo "Set QWEN14_SOURCE_DATA_DIR to the prior server_outputs/data directory," >&2
    echo "or set QWEN14_AUCTION_CSV to the licensed Online Auctions auction.csv." >&2
    exit 1
  fi
fi

export PYTHONUNBUFFERED=1
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_qwen3_14b.py" --output-dir "${RESULTS_DIR}" --stage inputs

case "${MODE}" in
  deterministic|no_think|stochastic|inference|all)
    "${PYTHON_BIN}" "${SCRIPT_DIR}/check_environment.py"
    ;;
esac

COST_ARGS=()
if [[ -n "${ROBUSTNESS_GPU_HOUR_USD:-}" ]]; then
  COST_ARGS+=(--gpu-hour-usd "${ROBUSTNESS_GPU_HOUR_USD}")
fi
if [[ -n "${ROBUSTNESS_ELECTRICITY_USD_PER_KWH:-}" ]]; then
  COST_ARGS+=(--electricity-usd-per-kwh "${ROBUSTNESS_ELECTRICITY_USD_PER_KWH}")
fi

run_deterministic() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_fixed_budget.py" \
    --config "${CONFIG_PATH}" --output-dir "${RESULTS_DIR}" --model qwen3_14b \
    --chat-mode reasoning --decoding greedy \
    --budgets 0,64,128,256,512,768,unconstrained --repeat-seeds 11,23,37 \
    --constrained-option-readout "${COST_ARGS[@]}" \
    2>&1 | tee -a "${RESULTS_DIR}/logs/qwen3_14b_deterministic.log"
}

run_no_think() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_fixed_budget.py" \
    --config "${CONFIG_PATH}" --output-dir "${RESULTS_DIR}" --model qwen3_14b \
    --chat-mode no_think --decoding greedy --budgets 0 --repeat-seeds 11,23,37 \
    --constrained-option-readout "${COST_ARGS[@]}" \
    2>&1 | tee -a "${RESULTS_DIR}/logs/qwen3_14b_no_think.log"
}

run_stochastic() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_fixed_budget.py" \
    --config "${CONFIG_PATH}" --output-dir "${RESULTS_DIR}" --model qwen3_14b \
    --chat-mode reasoning --decoding stochastic --budgets 128,384,768 \
    --repeat-seeds 1,7,21,42,99 --option-order-seed 314159 \
    --query-verbal-confidence --constrained-option-readout "${COST_ARGS[@]}" \
    2>&1 | tee -a "${RESULTS_DIR}/logs/qwen3_14b_stochastic.log"
}

run_analysis() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_fixed.py" --config "${CONFIG_PATH}" --output-dir "${RESULTS_DIR}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_extensions.py" --output-dir "${RESULTS_DIR}" --export-dir "${RESULTS_DIR}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_confidence_uncertainty.py" --output-dir "${RESULTS_DIR}" --export-dir "${RESULTS_DIR}" --bootstrap-samples 1000
  "${PYTHON_BIN}" "${SCRIPT_DIR}/bootstrap_stochastic_game.py" --output-dir "${RESULTS_DIR}" --samples 1000 --seed 20260819
  "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_freeform_sensitivity.py" --results-root "${RESULTS_DIR}" --bootstrap-samples 2000 --seed 20260822
  "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_submission_robustness.py" --output-dir "${RESULTS_DIR}" --samples 1000 --auction-samples 2000 --seed 20260822
  "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_normalized_game.py" --output-dir "${RESULTS_DIR}" --grid-points 181
  "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_zero_compute_augmented.py" --output-dir "${RESULTS_DIR}" --samples 1000 --seed 20260819 --grid-points 181
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_outputs.py" --config "${CONFIG_PATH}" --output-dir "${RESULTS_DIR}" --allow-unpriced --skip-direct
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_qwen3_14b.py" --output-dir "${RESULTS_DIR}" --stage complete
  "${PYTHON_BIN}" "${SCRIPT_DIR}/package_results.py" --output-dir "${RESULTS_DIR}" --zip qwen3_14b_paper_results.zip
}

case "${MODE}" in
  deterministic) run_deterministic ;;
  no_think) run_no_think ;;
  stochastic) run_stochastic ;;
  inference) run_deterministic; run_no_think; run_stochastic ;;
  analyze) run_analysis ;;
  validate) "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_qwen3_14b.py" --output-dir "${RESULTS_DIR}" --stage complete ;;
  estimate) "${PYTHON_BIN}" "${SCRIPT_DIR}/estimate_runtime.py" --output-dir "${RESULTS_DIR}" ;;
  all) run_deterministic; run_no_think; run_stochastic; run_analysis ;;
  *)
    echo "Usage: $0 {deterministic|no_think|stochastic|inference|analyze|validate|estimate|all}" >&2
    exit 2
    ;;
esac

echo "Qwen3-14B ${MODE} stage complete. Results: ${RESULTS_DIR}"
