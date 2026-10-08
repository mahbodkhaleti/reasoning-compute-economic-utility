#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${REPRO_PYTHON_BIN:-python3}"
MODE="${1:-full}"
if [[ "$MODE" != "full" && "$MODE" != "--use-existing-replay" ]]; then
  echo 'Usage: bash reproduce_cpu_revision.sh [--use-existing-replay]' >&2
  exit 2
fi
export MPLBACKEND=Agg
export MPLCONFIGDIR="${ROOT}/.cache/matplotlib"
export XDG_CACHE_HOME="${ROOT}/.cache"
export PYTHONHASHSEED=0
export REPRO_PYTHON_BIN="$PYTHON_BIN"
mkdir -p "$MPLCONFIGDIR"
"$PYTHON_BIN" "$ROOT/cpu_revision/verify_package.py"
if [[ "$MODE" == "full" ]]; then
  bash "$ROOT/reproduce_all_from_preserved.sh"
else
  "$PYTHON_BIN" "$ROOT/offline_replay/verify_preserved.py" --repo-root "$ROOT"
fi
"$PYTHON_BIN" "$ROOT/cpu_revision/analyze_legacy_revision.py" --snapshot "$ROOT" --out "$ROOT/cpu_revision/legacy_results" --draws 10000
"$PYTHON_BIN" "$ROOT/cpu_revision/analyze_cpu_extensions.py" --root "$ROOT" --draws 10000 --seed 20261008
"$PYTHON_BIN" "$ROOT/cpu_revision/export_extensions.py"
"$PYTHON_BIN" "$ROOT/cpu_revision/validate_cpu_revision.py"
"$PYTHON_BIN" "$ROOT/decision_audit/analyze_saved_decisions.py"
"$PYTHON_BIN" "$ROOT/decision_audit/export_audit.py"
"$PYTHON_BIN" "$ROOT/decision_audit/validate_audit.py"
mkdir -p "$ROOT/decision_audit/human_study/responses"
set +e
"$PYTHON_BIN" "$ROOT/decision_audit/human_study/analyze_human_choices.py" --responses "$ROOT/decision_audit/human_study/responses"
human_analysis_exit=$?
set -e
if [[ "$human_analysis_exit" != 0 && "$human_analysis_exit" != 3 ]]; then
  exit "$human_analysis_exit"
fi
echo 'CPU analyses completed. Check the human-study status separately; pending responses do not pass its evidence gate.'
