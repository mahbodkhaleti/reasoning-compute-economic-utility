#!/usr/bin/env bash
# Smoke test for Run 1 (about 30-60 min, mostly model downloads).
# 1) drift check: paper protocol at 768 on 10 scenarios x 1 order, compared with the paper's traces
# 2) Run 1 itself on 10 scenarios x 1 order, compared with the paper's 1,024-cap traces
set -uo pipefail
cd "$(dirname "$0")"; source env.sh
mkdir -p logs
for m in qwen3_4b qwen3_8b deepseek_r1_llama_8b qwen3_14b; do
  $(pyfor $m) followup/run_followup.py --run drift768 --model $m --smoke 2>&1 | tee logs/smoke_drift768_$m.log
  $(pyfor $m) followup/run_followup.py --run run1 --model $m --cap 4096 --smoke 2>&1 | tee logs/smoke_run1_$m.log
done
echo; echo "===== DRIFT CHECK (forced 768) ====="
env_a/bin/python followup/compare_drift.py results/followup/smoke/drift768 | tee logs/smoke_compare_drift768.txt
echo; echo "===== RUN 1 PREFIX CHECK ====="
env_a/bin/python followup/compare_drift.py results/followup/smoke/run1_nat4096 | tee logs/smoke_compare_run1.txt
echo; echo "===== RUN 1 SMOKE SUMMARY ====="
env_a/bin/python followup/quick_summary.py results/followup/smoke/run1_nat4096 | tee logs/smoke_summary_run1.txt
tar czf smoke_run1_results.tgz results/followup/smoke logs
echo "Send me smoke_run1_results.tgz"
