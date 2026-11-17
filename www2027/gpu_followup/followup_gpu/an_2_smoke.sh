#!/usr/bin/env bash
# Smoke (2 scenarios x 1 order, Qwen3-4B), R1 smoke, prefix-reuse test (2 models x 20), and the HF<->vLLM
# equivalence check (6 models x 40 eBay decisions at 768). Prints a cost estimate. About 1-2 h, mostly downloads.
set -uo pipefail
cd "$(dirname "$0")"; source env.sh; mkdir -p logs
R="env_a/bin/python -m auctionnet_runner.run"
$R run --dataset auctionnet --model qwen3_4b --run-type forced --smoke 2>&1 | tee logs/an_smoke_forced.log
$R run --dataset auctionnet --model qwen3_4b --run-type unforced1024 --smoke 2>&1 | tee logs/an_smoke_unforced.log
$R run --dataset ebay --model phi4_mini_reasoning --run-type readout_ablation --smoke 2>&1 | tee logs/an_smoke_r1.log
for m in qwen3_4b deepseek_r1_llama_8b; do $R run --dataset ebay --model $m --run-type prefix_test 2>&1 | tee -a logs/an_prefix.log; done
for m in phi4_mini_reasoning qwen3_4b qwen3_8b deepseek_r1_llama_8b granite_3_2_8b_instruct qwen3_14b; do
  $R run --dataset ebay --model $m --run-type equiv 2>&1 | tee -a logs/an_equiv.log; done
$R estimate | tee logs/an_estimate.log
grep -h "EQUIVALENCE\|PREFIX" logs/an_equiv.log logs/an_prefix.log
tar czf an_smoke_results.tgz results/smoke results/extra logs/an_*.log
echo "Send me an_smoke_results.tgz (do not start an_3_p0.sh before my OK)"
