#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/src/verify_synthetic.py" --data-dir "$ROOT/data" --output-dir "$ROOT/results" --models phi4_mini_reasoning,qwen3_4b,deepseek_r1_llama_8b,qwen3_8b
