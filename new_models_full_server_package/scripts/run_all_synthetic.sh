#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; bash "$ROOT/scripts/run_qwen3_14b_synthetic.sh"; bash "$ROOT/scripts/run_granite_synthetic.sh"
