#!/usr/bin/env bash
# The analysis scripts in www2027/ use the fixed root /tmp/www2027.
# This script points that root at this folder and links the raw eBay model traces,
# which are already stored elsewhere in this repository, into www2027/raw_traces.
# Run once from anywhere:  bash www2027/setup_paths.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ln -sfn "$HERE" /tmp/www2027
mkdir -p "$HERE/raw_traces"
link() { ln -sfn "$REPO/$1" "$HERE/raw_traces/$(basename "$1")"; }
link robustness_extensions/server_outputs/raw/fixed_deepseek_r1_llama_8b_reasoning_greedy.jsonl
link robustness_extensions/server_outputs/raw/fixed_phi4_mini_reasoning_reasoning_greedy.jsonl
link robustness_extensions/server_outputs/raw/fixed_qwen3_8b_reasoning_greedy.jsonl
link new_models_full_server_package/data/existing4_results/fixed_qwen3_4b_reasoning_greedy.jsonl
link preserved_outputs/qwen3_14b/historical/raw/fixed_qwen3_14b_reasoning_greedy.jsonl
link preserved_outputs/granite_3_2_8b_instruct/historical/raw/fixed_granite_3_2_8b_instruct_reasoning_greedy.jsonl
echo "/tmp/www2027 -> $HERE"
echo "raw_traces linked: $(ls "$HERE/raw_traces" | wc -l) files"
