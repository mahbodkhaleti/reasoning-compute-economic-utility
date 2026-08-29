#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?usage: run_model_full.sh MODEL_SLUG}"
bash "$ROOT/scripts/run_historical_model.sh" "$MODEL"
bash "$ROOT/scripts/run_synthetic_model.sh" "$MODEL"
