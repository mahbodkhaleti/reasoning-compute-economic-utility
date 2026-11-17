#!/usr/bin/env bash
# Full Run 1 (216 scenarios x 3 orders per model, cap 4,096). Resumable: just rerun.
# Two streams share the 48 GB GPU while the 8B-class models run; Qwen3-14B runs alone after.
set -uo pipefail
cd "$(dirname "$0")"; source env.sh
mkdir -p logs
( for m in qwen3_4b qwen3_8b; do $(pyfor $m) followup/run_followup.py --run run1 --model $m --cap 4096 >> logs/run1_$m.log 2>&1; done ) &
( $(pyfor deepseek_r1_llama_8b) followup/run_followup.py --run run1 --model deepseek_r1_llama_8b --cap 4096 >> logs/run1_deepseek_r1_llama_8b.log 2>&1 ) &
wait
$(pyfor qwen3_14b) followup/run_followup.py --run run1 --model qwen3_14b --cap 4096 >> logs/run1_qwen3_14b.log 2>&1
env_a/bin/python followup/quick_summary.py results/followup/run1_nat4096 | tee logs/summary_run1.txt
tar czf run1_results.tgz results/followup/run1_nat4096 logs
echo "Run 1 done. Send me run1_results.tgz"
