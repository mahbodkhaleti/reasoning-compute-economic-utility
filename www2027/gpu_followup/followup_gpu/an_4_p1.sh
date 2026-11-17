#!/usr/bin/env bash
# P1: AuctionNet unforced, cap 4,096 (A3) for the four models of the eBay 4,096 run. Resumable.
set -uo pipefail
cd "$(dirname "$0")"; source env.sh; mkdir -p logs
for m in qwen3_4b qwen3_8b deepseek_r1_llama_8b qwen3_14b; do
  env_a/bin/python -m auctionnet_runner.run run --dataset auctionnet --model $m --run-type unforced4096 >> logs/an_a3_$m.log 2>&1; done
tar czf an_p1_results.tgz results/auctionnet/decisions/*unforced4096* results/auctionnet/runlog.jsonl logs/an_a3_*.log
echo "P1 done. Send me an_p1_results.tgz"
