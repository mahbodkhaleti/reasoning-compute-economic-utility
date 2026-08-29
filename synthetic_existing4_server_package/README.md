# Synthetic IPV benchmark — existing four models

This standalone package runs only the canonical `synthetic-ipv-fpsb-v1.0.0` robustness benchmark for the four already-published checkpoints. It does **not** rerun their historical-auction experiments.

The checked-in manifest contains 216 scenarios: K=5/K=7 crossed with structured/unstructured candidates, 54 scenarios per cell. Values are stratified from 8 to 99. Opponent values are independent discrete uniforms on `{0,...,99}`, opponents bid `2v/3`, and exact win probabilities enumerate all 10,000 opponent-value pairs with equal splitting of highest-bid ties. Candidate sets are generated before utilities are evaluated and are not conditioned on the optimum. See `data/benchmark_metadata.json` and `data/manifest.sha256`.

## Setup and run

```bash
cd synthetic_existing4_server_package
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=...
bash scripts/smoke_test.sh phi4_mini_reasoning
bash scripts/run_all.sh
bash scripts/verify_results.sh
bash scripts/analyze.sh
bash scripts/make_paper_outputs.sh
```

Individual runners are `run_phi_mini.sh`, `run_qwen3_4b.sh`, `run_deepseek_8b.sh`, and `run_qwen3_8b.sh`. `smoke_test_all_models.sh` runs the isolated tiny test on all four. Each model has `216 × 7 × 3 = 4,536` expected records; the package total is 18,144. Results are written after every cell and a repeated identical command skips complete keys.

Set cost and hardware fields in `config/experiment.json` before inference if monetary reporting is needed. BF16 is the default and quantization is never enabled automatically. Typical single-GPU recommendations are about 16 GB for the small checkpoints and 24 GB for the 8B checkpoints, with additional headroom depending on attention implementation and context cache.

For a disconnected SSH-safe run:

```bash
tmux new -s synthetic4
source .venv/bin/activate
export HF_TOKEN=...
bash scripts/run_all_synthetic.sh
# Ctrl-b d; later: tmux attach -t synthetic4
tail -f results/logs/qwen3_8b.log
```

GPU is required only by `run_model.sh`, individual model wrappers, `run_all_synthetic.sh`, and `smoke_test.sh`. Verification and every analysis are CPU-only.
