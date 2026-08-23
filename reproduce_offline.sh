#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="${ROOT}/robustness_extensions/scripts"
OUTPUTS="${ROOT}/robustness_extensions/server_outputs"
CONFIG="${SCRIPTS}/config.json"
PYTHON_BIN="${REPRO_PYTHON_BIN:-python3}"

# analyze_fixed.py historically derives some CI seeds from Python hash(). This
# fixes process hash randomization for prospective reruns. It cannot recover an
# unrecorded historical PYTHONHASHSEED.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

"${PYTHON_BIN}" "${SCRIPTS}/analyze_fixed.py" \
  --config "${CONFIG}" --output-dir "${OUTPUTS}"
"${PYTHON_BIN}" "${SCRIPTS}/analyze_extensions.py" \
  --output-dir "${OUTPUTS}" --export-dir "${OUTPUTS}" --bootstrap-samples 2000
"${PYTHON_BIN}" "${SCRIPTS}/analyze_confidence_uncertainty.py" \
  --output-dir "${OUTPUTS}" --export-dir "${OUTPUTS}" --bootstrap-samples 1000
"${PYTHON_BIN}" "${SCRIPTS}/bootstrap_stochastic_game.py" \
  --output-dir "${OUTPUTS}" --samples 1000 --seed 20260819
"${PYTHON_BIN}" "${SCRIPTS}/analyze_freeform_sensitivity.py" \
  --results-root "${OUTPUTS}" --bootstrap-samples 2000 --seed 20260822
"${PYTHON_BIN}" "${SCRIPTS}/analyze_submission_robustness.py" \
  --output-dir "${OUTPUTS}" --samples 1000 --auction-samples 2000 --seed 20260822
"${PYTHON_BIN}" "${SCRIPTS}/analyze_normalized_game.py" \
  --output-dir "${OUTPUTS}" --grid-points 181
"${PYTHON_BIN}" "${SCRIPTS}/analyze_zero_compute_augmented.py" \
  --output-dir "${OUTPUTS}" --samples 1000 --seed 20260819 --grid-points 181

"${PYTHON_BIN}" "${ROOT}/validate_repository.py"

echo "Offline analyses and repository validation completed."
