#!/usr/bin/env bash
set -euo pipefail
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
TASK_PYTHON="${REPRO_PYTHON_BIN:-python3}"
"$TASK_PYTHON" "$TASK_REPO_ROOT/ai4f_workshop/verify_release.py"
"$TASK_PYTHON" "$TASK_REPO_ROOT/ai4f_workshop/check_method_bounds.py"
"$TASK_PYTHON" "$TASK_REPO_ROOT/ai4f_workshop/cpu_reference/test_reference.py"
"$TASK_PYTHON" "$TASK_REPO_ROOT/ai4f_workshop/cpu_reference/check_saved_protocol.py"
"$TASK_PYTHON" "$TASK_REPO_ROOT/ai4f_workshop/verify_numbers.py"
