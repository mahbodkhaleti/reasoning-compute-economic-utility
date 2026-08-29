#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for model in phi4_mini_reasoning qwen3_4b deepseek_r1_llama_8b qwen3_8b; do
  bash "$ROOT/scripts/run_model.sh" "$model"
done
