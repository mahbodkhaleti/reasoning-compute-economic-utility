#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_DIR}/robustness_extensions/server_outputs"
CONFIG_PATH="${SCRIPT_DIR}/config.json"
DEFAULT_PYTHON_BIN="${PROJECT_DIR}/robustness_extensions/.venv/bin/python"
if [[ -n "${ROBUSTNESS_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${ROBUSTNESS_PYTHON_BIN}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
fi
AUCTION_CSV="${PROJECT_DIR}/online-auctions-dataset/auction.csv"

ROBUSTNESS_GPU_HOUR_USD="${ROBUSTNESS_GPU_HOUR_USD:-}"
ROBUSTNESS_ELECTRICITY_USD_PER_KWH="${ROBUSTNESS_ELECTRICITY_USD_PER_KWH:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  python3 -m venv "${PROJECT_DIR}/robustness_extensions/.venv"
  PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
fi
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r "${SCRIPT_DIR}/requirements.txt"

COMMON_COST=()
if [[ -n "${ROBUSTNESS_GPU_HOUR_USD}" ]]; then
  COMMON_COST+=(--gpu-hour-usd "${ROBUSTNESS_GPU_HOUR_USD}")
fi
if [[ -n "${ROBUSTNESS_ELECTRICITY_USD_PER_KWH}" ]]; then
  COMMON_COST+=(--electricity-usd-per-kwh "${ROBUSTNESS_ELECTRICITY_USD_PER_KWH}")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_scenarios.py" \
  --config "${CONFIG_PATH}" --auction-csv "${AUCTION_CSV}" --output-dir "${OUTPUT_DIR}"

MODELS=(phi4_mini_reasoning qwen3_4b deepseek_r1_llama_8b qwen3_8b)
for MODEL in "${MODELS[@]}"; do
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_fixed_budget.py" \
    --config "${CONFIG_PATH}" --output-dir "${OUTPUT_DIR}" --model "${MODEL}" \
    --chat-mode reasoning --decoding greedy \
    --budgets 0,64,128,256,512,768,unconstrained --repeat-seeds 11,23,37 \
    --constrained-option-readout \
    "${COMMON_COST[@]}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_fixed_budget.py" \
    --config "${CONFIG_PATH}" --output-dir "${OUTPUT_DIR}" --model "${MODEL}" \
    --chat-mode no_think --decoding greedy --budgets 0 --repeat-seeds 11,23,37 \
    --constrained-option-readout \
    "${COMMON_COST[@]}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_fixed_budget.py" \
    --config "${CONFIG_PATH}" --output-dir "${OUTPUT_DIR}" --model "${MODEL}" \
    --chat-mode reasoning --decoding stochastic --budgets 128,384,768 \
    --repeat-seeds 1,7,21,42,99 --option-order-seed 314159 \
    --query-verbal-confidence --constrained-option-readout \
    "${COMMON_COST[@]}"
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_fixed.py" --config "${CONFIG_PATH}" --output-dir "${OUTPUT_DIR}"

if [[ -n "${ROBUSTNESS_GPU_HOUR_USD}" ]]; then
  for MODEL in "${MODELS[@]}"; do
    for MODE in adaptive fixed_low fixed_high; do
      "${PYTHON_BIN}" "${SCRIPT_DIR}/run_direct_game.py" \
        --config "${CONFIG_PATH}" --output-dir "${OUTPUT_DIR}" --model "${MODEL}" \
        --mode "${MODE}" "${COMMON_COST[@]}"
    done
  done
  "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_direct.py" --config "${CONFIG_PATH}" --output-dir "${OUTPUT_DIR}"
  VALIDATION_FLAGS=()
else
  echo "ROBUSTNESS_GPU_HOUR_USD is unset: dollar-cost calibration and price-dependent direct games will be skipped."
  VALIDATION_FLAGS=(--allow-unpriced --skip-direct)
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_extensions.py" \
  --output-dir "${OUTPUT_DIR}" --export-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_confidence_uncertainty.py" \
  --output-dir "${OUTPUT_DIR}" --export-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/bootstrap_stochastic_game.py" \
  --output-dir "${OUTPUT_DIR}" --samples 1000
"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_freeform_sensitivity.py" \
  --results-root "${OUTPUT_DIR}" --bootstrap-samples 2000 --seed 20260822
"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_submission_robustness.py" \
  --output-dir "${OUTPUT_DIR}" --samples 1000 --auction-samples 2000 --seed 20260822
"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_normalized_game.py" \
  --output-dir "${OUTPUT_DIR}" --grid-points 181
"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_zero_compute_augmented.py" \
  --output-dir "${OUTPUT_DIR}" --samples 1000 --seed 20260819 --grid-points 181
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_outputs.py" \
  --config "${CONFIG_PATH}" --output-dir "${OUTPUT_DIR}" "${VALIDATION_FLAGS[@]}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/package_results.py" \
  --output-dir "${OUTPUT_DIR}" --zip robustness_4090_results.zip

echo "Complete. Reproducibility archive: robustness_extensions/scripts/robustness_4090_results.zip"
