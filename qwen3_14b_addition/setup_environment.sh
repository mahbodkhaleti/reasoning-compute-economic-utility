#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${QWEN14_VENV_DIR:-${PACKAGE_DIR}/.venv}"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${PACKAGE_DIR}/requirements.txt"
"${VENV_DIR}/bin/python" "${PACKAGE_DIR}/scripts/check_environment.py"

echo "Environment ready. Use:"
echo "export QWEN14_PYTHON_BIN=${VENV_DIR}/bin/python"
