# Budget, Readout or Reasoning? Where Small Reasoning Models Lose Value in Auction Bidding

This folder holds the code, saved decisions and analysis outputs for the short paper submitted to the WWW 2027 research track (Web Economics and Digital Society). The anonymous paper and its LaTeX source are in [`paper/`](paper/).

Six open reasoning models (Phi-4-mini-reasoning, Qwen3-4B, -8B and -14B, DeepSeek-R1-Distill-Llama-8B and Granite-3.2-8B-Instruct) choose a bid from a bid landscape. Win curves come from eBay closing prices and from AuctionNet, a simulated advertising market. Bids are scored on held-out auctions against an exact calculator, and every reasoning trace is parsed by code to find where the gap to the calculator opens.

## Quick start

No GPU is needed to reproduce the numbers in the paper from the saved decisions.

```sh
bash www2027/setup_paths.sh          # points /tmp/www2027 at this folder, links raw eBay traces
pip install numpy pandas scipy statsmodels matplotlib
cd /tmp/www2027/analysis/rev25 && python ../followup/econ_surplus_pct.py
cd /tmp/www2027/analysis/auctionnet && python an_overnight_analysis.py
cd /tmp/www2027/analysis/arxiv && python make_tables.py
```

The scripts use the fixed root `/tmp/www2027`, which `setup_paths.sh` creates as a link to this folder. Scripts in `analysis/rev25/` and `analysis/followup/` import `r_common.py`, so run them from `analysis/rev25/`. Each script writes to `results/` and does not change the decision files.

## Where each result comes from

| Paper item | Script (under `analysis/`) | Output (under `results/` unless noted) |
| --- | --- | --- |
| Table 1: $q(0)$, $q(768)$, $q$ by budget | `arxiv/collect_ebay.py` | `arxiv/ebay_budget.json` |
| Table 1: Answer (768 tokens) and dollar recovery | `revision/t10_judge7.py` (B8) | `analysis/revision/out/t10_judge7.json` |
| Table 1: Closed (closed-block readout) | `followup/r1_analysis.py` | `R1/r1_summary.json` |
| Table 1: Stated answer | `rev25/r3_stated_answer.py` | `rev25/r3_stated_answer.json` |
| Table 1: 1,024 run | `revision/t8_fixes.py` (F1) | `analysis/revision/out/t8_fixes.json` |
| Table 1: FP eq. | `rev25/r1_firstprice.py` | `rev25/r1_firstprice.json` |
| Table 1: AuctionNet | `auctionnet/an_p0_analysis.py` | `AN/an_p0_summary.json` |
| Table 1: Break-even, cost | `rev25/r6_r8_robustness.py` | `rev25/r6_r8_robustness.json` |
| Table 2: gap split | `revision/t10b_hybrid_split.py` | `analysis/revision/out/t10b_hybrid_split.json` |
| Table 2: finished, medians; Figure 1b | `revision/t9_judge6.py` | `analysis/revision/out/t9_judge6.json` |
| Arithmetic upper bound | `rev25/r2b_reader_only_split.py` | `rev25/r2b_reader_only_split.json` |
| Figure 1 | `rev25/fig_main_v43.py` | `paper/figures/main_figure_v43.pdf` |
| Unforced 4,096-token runs | `followup/run1_analysis.py`, `followup/run1_phi4096_analysis.py` | `followup/run1_summary.json`, `followup/run1_phi4096_summary.json` |
| Prompting (list instruction) | `followup/run4_analysis.py` | `followup/run4_summary.json` |
| Closed-block readout, downstream analyses | `followup/r1_closed_downstream.py` | `R1/downstream_open.json`, `R1/downstream_closed.json` |
| Highest-bid share, surplus loss | `followup/econ_surplus_pct.py` | `followup/econ_surplus_pct.json` |
| Size versus position | `followup/position_vs_size.py` | `position_size/position_vs_size.json` |
| Lean under sampling | `followup/sampled_highest.py` | `followup/sampled_highest.json` |
| Passing-over regression | `rev25/r5_passover_regression.py` | `rev25/r5_passover_regression.json` |
| Audit, out of sample | `rev25/r4_audit_oos.py` | `rev25/r4_audit_oos.json` |
| Product-level and landscape bootstrap | `rev25/r6_r8_robustness.py` | `rev25/r6_r8_robustness.json` |
| First-price variants | `rev25/r9_fp_robustness.py` | `rev25/r9_fp_robustness.json` |
| AuctionNet gap split | `auctionnet/an_split.py` | `AN/an_split.json`, `AN/an_split_closed.json` |
| AuctionNet regression and audit | `auctionnet/an_regression_audit.py` | `AN/an_regression_audit.json` |
| AuctionNet closed readout, 4,096 runs | `auctionnet/an_overnight_analysis.py` | `AN/an_overnight_summary.json` |
| Extended tables (all of the above, as LaTeX) | `arxiv/make_tables.py`, `arxiv/fig_auctionnet.py` | `arxiv/tables/`, `arxiv/figures/` |

Trace parsing uses the matcher `analysis/extract5.py` and the text reader `analysis/revision/extract8.py`. A summary of the AuctionNet results is in `results/AN/AN_P0_SUMMARY.md`.

## Saved decisions

| Run | Location |
| --- | --- |
| eBay forced budgets, HF transformers (original runs) | elsewhere in this repository; `setup_paths.sh` links them into `raw_traces/` |
| eBay unforced 4,096-token runs | `results/followup/results/followup/run1_nat4096/`, `results/followup/phi4096/` |
| eBay prompting variant | `results/followup/run4/` |
| eBay closed-block readout | `results/R1/results/extra/R1_readout_ablation/` |
| eBay vLLM check (Phi-mini, Granite-8B) | `results/AN/ov/results/extra/ebay_vllm/` |
| AuctionNet forced budgets and unforced runs | `results/AN/ov/results/auctionnet/decisions/` |
| AuctionNet closed-block readout | `results/AN/ov/results/auctionnet/R1_readout_ablation/` |
| AuctionNet scenarios | `results/AN/ov/results/auctionnet/scenarios.csv` |

Each decision row records the scenario, option order, reasoning text, chosen bid, letter probabilities and held-out utility.

## Rerunning the models

`gpu_followup/followup_gpu/` contains the runners used for the GPU runs, with setup scripts for a single rented GPU.

- `followup/` and `pipeline/`: the eBay runs (unforced 4,096 tokens, prompting variant, closed-block readout), reusing the original pipeline code.
- `auctionnet_runner/`: builds AuctionNet scenarios from the raw data and runs all models with vLLM. `an_1_data.sh` to `an_5_overnight.sh` run the steps in order.
- `docs/PIPELINE_MAP.md` explains how the code maps to the original pipeline, and `docs/DATA_LICENSE.md` covers AuctionNet's licence.

## Data and licences

- **eBay.** The auction data come from the public research dataset accompanying *Modeling Online Auctions* (Jank and Shmueli), in its Kaggle copy. The raw file is not redistributed here; download it from the source. Scenarios use only prices.
- **AuctionNet.** AuctionNet is simulated data distributed under Apache 2.0 (Su et al., NeurIPS 2024 Datasets and Benchmarks). The raw files are not committed; `auctionnet_runner/data.py` downloads them. Only derived scenario files and decisions are included, with attribution.
- No model weights are included.

## Other folders

- `judge/`: inputs and outputs of internal review checks used while revising the paper.
- `human_annotation/`: a package for hand-labelling 100 traces. It was not used for any number in the paper; all splits come from code.
