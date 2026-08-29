#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; bash "$ROOT/scripts/smoke_test.sh" qwen3_14b
