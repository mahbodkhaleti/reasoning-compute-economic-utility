#!/usr/bin/env bash
# Overnight queue, started once. Waits for P0 and the queued eBay vLLM check to finish, then:
#  1. AuctionNet closed-block readout (R1) on the saved 768-token forced traces, all 6 models (prefill only).
#     A 2-decision smoke run first checks that the recomputed original letter matches the saved one.
#  2. P1: AuctionNet unforced, cap 4,096, for Qwen3-4B, Qwen3-8B, DeepSeek-8B, Qwen3-14B.
# Packs results after each step. Resumable: rerunning skips finished decisions.
set -uo pipefail
cd "$(dirname "$0")"; source env.sh; mkdir -p logs
R="env_a/bin/python -m auctionnet_runner.run"
echo "$(date -u) waiting for P0 and the eBay check" >> logs/an_overnight.log
while pgrep -f "[a]n_3_p0.sh" >/dev/null || pgrep -f "[w]hile pgrep -f" >/dev/null || pgrep -f "[d]ataset ebay --model" >/dev/null; do sleep 60; done
echo "$(date -u) P0 and eBay check finished" >> logs/an_overnight.log
tar czf an_p0_ebay_results.tgz results/auctionnet results/extra logs/an_*.log logs/ebay_vllm.out 2>/dev/null
tar czf ebay_vllm_results.tgz results/extra/ebay_vllm logs/ebay_vllm.out 2>/dev/null
# 1. AuctionNet closed-block readout
$R run --dataset auctionnet --model qwen3_4b --run-type readout_ablation --smoke >> logs/an_r1an.log 2>&1
SMOKE_OK=$(env_a/bin/python - <<'PY'
import json
rows=[json.loads(l) for l in open("results/smoke/auctionnet/R1_readout_ablation/qwen3_4b.jsonl")]
print("yes" if rows and all(r["orig_label"]==r["saved_label"] for r in rows) else "no")
PY
)
echo "$(date -u) R1 AuctionNet smoke reproduces saved letters: $SMOKE_OK" >> logs/an_overnight.log
if [ "$SMOKE_OK" = "yes" ]; then
  for m in phi4_mini_reasoning granite_3_2_8b_instruct qwen3_4b qwen3_8b deepseek_r1_llama_8b qwen3_14b; do
    $R run --dataset auctionnet --model $m --run-type readout_ablation >> logs/an_r1an.log 2>&1
    echo "$(date -u) R1 AuctionNet $m exit $?" >> logs/an_overnight.log
  done
fi
tar czf an_r1an_results.tgz results/auctionnet/R1_readout_ablation results/smoke/auctionnet/R1_readout_ablation logs/an_r1an.log logs/an_overnight.log 2>/dev/null
# 2. P1
for m in qwen3_4b qwen3_8b deepseek_r1_llama_8b qwen3_14b; do
  $R run --dataset auctionnet --model $m --run-type unforced4096 >> logs/an_a3_$m.log 2>&1
  echo "$(date -u) P1 $m exit $?" >> logs/an_overnight.log
done
$R status | tee logs/an_status_overnight.log
tar czf an_overnight_results.tgz results/auctionnet results/extra logs/an_*.log logs/ebay_vllm.out 2>/dev/null
echo "$(date -u) ALL DONE. Send an_overnight_results.tgz" >> logs/an_overnight.log
