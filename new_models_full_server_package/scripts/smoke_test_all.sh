#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; bash "$ROOT/scripts/smoke_test_qwen3_14b.sh"; bash "$ROOT/scripts/smoke_test_granite.sh"
