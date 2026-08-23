#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_DIR="${PROJECT_DIR}/robustness_extensions/.analysis_venv"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  python3 -m venv "${ENV_DIR}"
  "${ENV_DIR}/bin/pip" install --upgrade pip
  "${ENV_DIR}/bin/pip" install numpy pandas matplotlib
fi

"${ENV_DIR}/bin/python" "${SCRIPT_DIR}/bootstrap_uncertainty.py" \
  --root "${PROJECT_DIR}" --output-dir robustness_extensions --samples 1000
