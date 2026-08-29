#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXISTING4="${1:-$ROOT/../synthetic_existing4_server_package}"
python "$ROOT/src/merge_six_synthetic.py" --package-root "$ROOT" --existing4-package "$EXISTING4"
