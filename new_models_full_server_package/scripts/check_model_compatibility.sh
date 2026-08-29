#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/src/inspect_model_compatibility.py" --config "$ROOT/config/experiment.json" --output "$ROOT/logs/model_compatibility.json"
