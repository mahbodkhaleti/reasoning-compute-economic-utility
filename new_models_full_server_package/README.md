# Qwen3-14B and Granite-3.2-8B full server package

This folder is transferable as a unit. It runs the exact saved 216-scenario historical benchmark and the canonical 216-scenario synthetic IPV robustness benchmark for two pinned checkpoints. It also carries checksum-tracked saved rows for the existing four models, used only in no-rerun combined figures and the Qwen-family scale sensitivity.

Pinned models:

- `Qwen/Qwen3-14B` @ `40c069824f4251a91eefaf281ebe4c544efd3e18`
- `ibm-granite/granite-3.2-8b-instruct` @ `610d8c6ee9c84ce51f6dfd7bc5c0215d95d49695`

Exact protocol details are in `docs/MODEL_RESOLUTION.json` and `docs/GRANITE_PROTOCOL_NOTE.md`. The Granite rows are machine-labelled as a protocol sensitivity.

## Setup and exact commands

```bash
cd new_models_full_server_package
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=...
```

Verify the two pinned tokenizers and their distinct template arguments before allocating a GPU:

```bash
bash scripts/check_model_compatibility.sh
```

Then run the smoke tests and workloads:

```bash
bash scripts/smoke_test_qwen3_14b.sh
bash scripts/smoke_test_granite.sh
bash scripts/run_qwen3_14b_historical.sh
bash scripts/run_granite_historical.sh
bash scripts/run_qwen3_14b_synthetic.sh
bash scripts/run_granite_synthetic.sh
bash scripts/verify_all.sh
bash scripts/analyze_all.sh
bash scripts/make_paper_outputs.sh
```

Convenience commands `run_all_historical.sh` and `run_all_synthetic.sh` run models sequentially, never concurrently. `run_all_additional_models.sh` performs both workloads sequentially. To merge the synthetic results with the companion package after both finish: `bash scripts/merge_six_model_synthetic.sh ../synthetic_existing4_server_package`.

Each model produces 3,888 historical deterministic records, 648 unconstrained records, 648 no-thinking records, 3,240 stochastic records, and 4,536 synthetic records: 12,960 total. The greedy file intentionally holds deterministic and unconstrained cells together, while verification reports them separately. Expected counts are derived from scenario manifests and cross-checked against these design totals.

All runners write one JSON object per completed cell, maintain invocation manifests, compact duplicate keys, reject mismatched resumptions, save exact resolved revisions before inference, and tolerate unavailable NVML metrics. No automatic quantization or precision fallback occurs. Actual hardware and a defensible hourly rate must be set in `config/experiment.json`; the default rate is `null`, so monetary lambda and kappa remain unavailable instead of applying the paper's single-4090 rate to different hardware.

## Memory and execution

Qwen3-14B has roughly 29.6 GB of BF16 weights; use at least a 40 GB GPU, with 48 GB recommended for safe generation headroom, or a supported multi-GPU `device_map="auto"` setup. Granite has roughly 16.3 GB of BF16 weights; a 24 GB GPU is recommended. Actual peak memory is saved per cell. If either model does not fit, the runner stops with a clear diagnostic; it will not silently quantize.

GPU-required scripts: the two model runners, `run_all_additional_models.sh`, and smoke tests. CPU-only after choices are saved: verification, all bootstraps, held-out CDF rescoring, strategic enumeration, figures, scale sensitivity, merge, and paper-table generation.

## Long remote runs

```bash
tmux new -s additional-models
source .venv/bin/activate
export HF_TOKEN=...
bash scripts/run_all_additional_models.sh
# detach: Ctrl-b d
tmux attach -t additional-models
tail -f logs/qwen3_14b_historical_deterministic.log
```

Smoke output is isolated under `smoke_results/` and cannot contaminate `results/`. Re-running the smoke command explicitly checks that resume adds no duplicate rows.

Primary manuscript outputs appear in `paper_ready/`. The synthetic benchmark is a robustness exercise inspired by the literature; it is not described as an exact replication of Zhu et al. or any prior benchmark.
