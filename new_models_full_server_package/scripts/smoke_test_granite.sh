#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; bash "$ROOT/scripts/smoke_test.sh" granite_3_2_8b_instruct
