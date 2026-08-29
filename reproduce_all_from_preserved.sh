#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPLAY="${ROOT}/offline_replay"
WORK="${REPLAY}/outputs/historical_six"
SYNTH="${REPLAY}/outputs/synthetic_six"
OLD_PANEL="${REPLAY}/outputs/reported_original_four"
QWEN_PANEL="${REPLAY}/outputs/reported_qwen3_14b"
GRANITE_PANEL="${REPLAY}/outputs/reported_granite_3_2_8b"
PYTHON_BIN="${REPRO_PYTHON_BIN:-python3}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONHASHSEED="0"

"${PYTHON_BIN}" "${REPLAY}/verify_preserved.py" --repo-root "${ROOT}"

mkdir -p "${WORK}/raw" "${WORK}/data" "${SYNTH}/raw" "${SYNTH}/paper_ready" \
  "${OLD_PANEL}/raw" "${OLD_PANEL}/data" "${QWEN_PANEL}/raw" "${QWEN_PANEL}/data" \
  "${GRANITE_PANEL}/raw" "${GRANITE_PANEL}/data"
cp -p "${ROOT}/robustness_extensions/server_outputs/raw/"fixed_*.jsonl "${WORK}/raw/"
cp -p "${ROOT}/robustness_extensions/server_outputs/raw/"fixed_*_metadata.json "${WORK}/raw/"
cp -p "${ROOT}/preserved_outputs/qwen3_14b/historical/raw/"fixed_*.jsonl "${WORK}/raw/"
cp -p "${ROOT}/preserved_outputs/qwen3_14b/historical/raw/"fixed_*_metadata.json "${WORK}/raw/"
cp -p "${ROOT}/preserved_outputs/granite_3_2_8b_instruct/historical/raw/"fixed_*.jsonl "${WORK}/raw/"
cp -p "${ROOT}/preserved_outputs/granite_3_2_8b_instruct/historical/raw/"fixed_*_metadata.json "${WORK}/raw/"
cp -p "${ROOT}/new_models_full_server_package/data/historical/"*.csv "${WORK}/data/"
cp -p "${ROOT}/robustness_extensions/server_outputs/raw/"fixed_*.jsonl "${OLD_PANEL}/raw/"
cp -p "${ROOT}/preserved_outputs/qwen3_14b/historical/raw/"fixed_*.jsonl "${QWEN_PANEL}/raw/"
# Granite's archived table was produced in the joint additional-models run.
# Stage Qwen here as well so the seeded RNG advances in the original order;
# the Granite rows then reproduce the reported finite-bootstrap endpoints.
cp -p "${ROOT}/preserved_outputs/qwen3_14b/historical/raw/"fixed_*.jsonl "${GRANITE_PANEL}/raw/"
cp -p "${ROOT}/preserved_outputs/granite_3_2_8b_instruct/historical/raw/"fixed_*.jsonl "${GRANITE_PANEL}/raw/"
cp -p "${ROOT}/new_models_full_server_package/data/historical/"*.csv "${OLD_PANEL}/data/"
cp -p "${ROOT}/new_models_full_server_package/data/historical/"*.csv "${QWEN_PANEL}/data/"
cp -p "${ROOT}/new_models_full_server_package/data/historical/"*.csv "${GRANITE_PANEL}/data/"
cp -p "${ROOT}/preserved_outputs/synthetic_existing4/raw/"synthetic_*_deterministic.jsonl "${SYNTH}/raw/"
cp -p "${ROOT}/preserved_outputs/qwen3_14b/synthetic/raw/"synthetic_*_deterministic.jsonl "${SYNTH}/raw/"
cp -p "${ROOT}/preserved_outputs/granite_3_2_8b_instruct/synthetic/raw/"synthetic_*_deterministic.jsonl "${SYNTH}/raw/"

"${PYTHON_BIN}" "${REPLAY}/src/analyze_fixed.py" --config "${REPLAY}/config/experiment.json" --output-dir "${WORK}"
"${PYTHON_BIN}" "${REPLAY}/src/analyze_extensions.py" --output-dir "${WORK}" --export-dir "${WORK}" --bootstrap-samples 2000
# The paper integrated the original-four panel, Qwen3-14B extension, and joint
# Qwen3-14B/Granite extension from separately seeded analysis runs. Replaying
# those panels separately preserves the reported bootstrap draw order exactly;
# the combined workspace is used for six-frontier plots and games.
"${PYTHON_BIN}" "${REPLAY}/src/analyze_extensions.py" --output-dir "${OLD_PANEL}" --export-dir "${OLD_PANEL}" --bootstrap-samples 2000
"${PYTHON_BIN}" "${REPLAY}/src/analyze_extensions.py" --output-dir "${QWEN_PANEL}" --export-dir "${QWEN_PANEL}" --bootstrap-samples 2000
"${PYTHON_BIN}" "${REPLAY}/src/analyze_extensions.py" --output-dir "${GRANITE_PANEL}" --export-dir "${GRANITE_PANEL}" --bootstrap-samples 2000
"${PYTHON_BIN}" "${REPLAY}/src/analyze_confidence_uncertainty.py" --output-dir "${WORK}" --export-dir "${WORK}" --bootstrap-samples 1000
"${PYTHON_BIN}" "${REPLAY}/src/bootstrap_stochastic_game.py" --output-dir "${WORK}" --samples 1000 --seed 20260819
"${PYTHON_BIN}" "${REPLAY}/src/analyze_freeform_sensitivity.py" --results-root "${WORK}" --bootstrap-samples 2000 --seed 20260822
"${PYTHON_BIN}" "${REPLAY}/src/analyze_submission_robustness.py" --output-dir "${WORK}" --samples 1000 --auction-samples 2000 --seed 20260822
"${PYTHON_BIN}" "${REPLAY}/src/analyze_normalized_game.py" --output-dir "${WORK}" --grid-points 181
"${PYTHON_BIN}" "${REPLAY}/src/analyze_zero_compute_augmented.py" --output-dir "${WORK}" --samples 1000 --seed 20260819 --grid-points 181
"${PYTHON_BIN}" "${REPLAY}/src/recompute_runtime_cost.py" --output-dir "${WORK}" --rtx4090-usd-per-hour 0.74

"${PYTHON_BIN}" "${REPLAY}/src/analyze_synthetic.py" --output-dir "${SYNTH}" --paper-ready "${SYNTH}/paper_ready" --bootstrap-samples 2000 --seed 20260824
"${PYTHON_BIN}" "${REPLAY}/src/analyze_synthetic_game.py" --output-dir "${SYNTH}" --grid-points 181

echo "Offline replay complete: ${REPLAY}/outputs"
