#!/usr/bin/env bash
# Run 4 smoke test: frees the Run 1-only models, then 10 scenarios x 1 order per model, both settings.
set -uo pipefail
cd "$(dirname "$0")"; source env.sh; mkdir -p logs
rm -rf /workspace/hf/hub/models--Qwen--Qwen3-4B /workspace/hf/hub/models--Qwen--Qwen3-14B
df -h /workspace | tail -1
for m in phi4_mini_reasoning granite_3_2_8b_instruct deepseek_r1_llama_8b qwen3_8b; do
  $(pyfor $m) followup/run_followup.py --run run4 --model $m --smoke 2>&1 | tee logs/smoke_run4_$m.log
done
echo; echo "===== RUN 4 SMOKE SUMMARY ====="
env_a/bin/python followup/quick_summary.py results/followup/smoke/run4_prompt | tee logs/smoke_summary_run4.txt
tar czf smoke_run4_results.tgz results/followup/smoke/run4_prompt logs/smoke_*run4*
echo "Send me smoke_run4_results.tgz"
