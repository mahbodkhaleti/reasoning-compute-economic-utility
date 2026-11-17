# GPU follow-up runs

Each run reuses the paper's own pipeline code, which is copied unchanged under `pipeline/`:

- `pipeline/orig` is `robustness_extensions/scripts`. It covers Phi-mini, Qwen3-4B, DeepSeek-8B and Qwen3-8B.
- `pipeline/new` is `new_models_full_server_package/src`. It covers Granite-8B and Qwen3-14B.

`followup/` adds only:

- the runner;
- prompt variants;
- a letter readout with labels as a parameter;
- checks.

Outputs go to `results/followup/<run_name>/<model>.jsonl`, one row per decision. Rerun any command to resume.

## Pod

- **GPU:** 1x RTX 6000 Ada (48 GB).
- **CUDA filter:** 13.0. The driver must be 580 or newer.
- **Template:** the RunPod PyTorch template.
- **Network volume:** about 150 GB, mounted at `/workspace`.
- **Environment variable:** optionally `HF_TOKEN`.

Upload `followup_gpu.zip` to `/workspace` (Jupyter upload works), then:

```bash
cd /workspace && unzip -o followup_gpu.zip && cd followup_gpu
bash setup_pod.sh                      # once, about 10 minutes
```

## Run 1: natural stopping, cap 4,096

```bash
bash smoke_run1.sh                     # drift check + 10-scenario smoke; send me smoke_run1_results.tgz
mkdir -p logs; nohup bash full_run1.sh > logs/full_run1.out 2>&1 &   # after my OK; survives closing the terminal
tail -f logs/run1_*.log                # progress with ETA
```

If a model has more than 10% of decisions at the cap, the 8,192 rerun covers only those decisions. A decision that stopped before 4,096 tokens stops the same way at 8,192, because greedy decoding is prefix-consistent.

```bash
source env.sh
$(pyfor qwen3_8b) followup/run_followup.py --run run1 --model qwen3_8b --cap 8192 \
    --only-capped-from results/followup/run1_nat4096/qwen3_8b.jsonl
```

## Later runs (scripts follow after Run 1)

```bash
source env.sh
$(pyfor MODEL) followup/run_followup.py --run run4  --model MODEL      # prompt instruction
env_a/bin/python followup/check_letters.py                                 # once, before Run 2a
$(pyfor MODEL) followup/run_followup.py --run run2a --model MODEL      # random letters
```

Add `--smoke` to any command to get 10 scenarios with order seed 11, in `results/followup/smoke/`.

## Checks built in

- **Library versions:** these must match the paper's run metadata, or the runner stops.
  - env_a: torch 2.13.0 and transformers 5.15.0.
  - env_b (Qwen3-14B): torch 2.9.1 and transformers 5.1.0.
- **Pinned revisions:** these come from the paper's config files.
- **Drift check:** `compare_drift.py` compares forced-768 traces with the paper's saved traces, token for token. It also checks that the new unforced traces start with the paper's 1,024-cap traces.
- **Energy:** measured on the GPU the process actually uses, through NVML.

## AuctionNet replication and extra runs

Run these on a separate pod: 1× RTX 6000 Ada with about **200 GB** of container disk.

`docs/PIPELINE_MAP.md` lists what is replicated from the eBay pipeline and where we deviate. `docs/DATA_LICENSE.md` covers the data licence and the fact that AuctionNet is simulated.

```bash
bash setup_an_pod.sh   # env_a / env_b (paper versions) + env_v (vLLM)
bash an_1_data.sh      # tests, download periods 7-10, win curves, 216 scenarios -> send an_data_check.tgz
bash an_2_smoke.sh     # smoke, R1 smoke, prefix test, HF<->vLLM equivalence on 40 eBay decisions per model -> send an_smoke_results.tgz
bash an_3_p0.sh        # after OK: R1 readout ablation (all 6), A1 forced + A2 unforced 1,024 (all 6)
bash an_4_p1.sh        # A3 unforced 4,096 (Qwen3-4B/8B/14B, DeepSeek-8B)
env_a/bin/python -m auctionnet_runner.run status
```

Every job is resumable. Each row is keyed by `(dataset, model, run_type, scenario_id, order_id, seed)`, so rerunning a script skips decisions that are already done.

Generation runs in vLLM from token ids only. Prompt token ids, decoding, the letter readout (`ModelHarness.constrained_bid_readout`), `parse_final_bid` and `score_bid` all run as unchanged eBay code, in the paper's library versions.
