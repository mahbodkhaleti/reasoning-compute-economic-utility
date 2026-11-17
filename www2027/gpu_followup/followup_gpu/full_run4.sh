#!/usr/bin/env bash
# Full Run 4 (prompt instruction): 4 models x 216 scenarios x 3 orders x {forced 768, unforced 1024}. Resumable.
set -uo pipefail
cd "$(dirname "$0")"; source env.sh; mkdir -p logs
( for m in phi4_mini_reasoning granite_3_2_8b_instruct; do $(pyfor $m) followup/run_followup.py --run run4 --model $m >> logs/run4_$m.log 2>&1; done ) &
( for m in deepseek_r1_llama_8b qwen3_8b; do $(pyfor $m) followup/run_followup.py --run run4 --model $m >> logs/run4_$m.log 2>&1; done ) &
wait
env_a/bin/python followup/quick_summary.py results/followup/run4_prompt | tee logs/summary_run4.txt
tar czf run4_results.tgz results/followup/run4_prompt logs/run4_* logs/summary_run4.txt
echo "Run 4 done. Send me run4_results.tgz"
