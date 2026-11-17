#!/usr/bin/env bash
# P0: R1 readout ablation (all 6, Phi-mini first), then AuctionNet forced (A1) and unforced 1,024 (A2) for all 6. Resumable.
set -uo pipefail
cd "$(dirname "$0")"; source env.sh; mkdir -p logs
R="env_a/bin/python -m auctionnet_runner.run"
for m in phi4_mini_reasoning granite_3_2_8b_instruct qwen3_4b qwen3_8b deepseek_r1_llama_8b qwen3_14b; do
  $R run --dataset ebay --model $m --run-type readout_ablation >> logs/an_r1.log 2>&1; done
tar czf an_r1_results.tgz results/extra/R1_readout_ablation logs/an_r1.log
for m in qwen3_4b qwen3_8b deepseek_r1_llama_8b phi4_mini_reasoning granite_3_2_8b_instruct qwen3_14b; do
  $R run --dataset auctionnet --model $m --run-type forced >> logs/an_a1_$m.log 2>&1
  $R run --dataset auctionnet --model $m --run-type unforced1024 >> logs/an_a2_$m.log 2>&1
done
$R status | tee logs/an_status_p0.log
tar czf an_p0_results.tgz results/auctionnet results/extra logs/an_*.log
echo "P0 done. Send me an_p0_results.tgz"
