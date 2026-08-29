# Reasoning compute and economic utility: reproducibility package

This repository contains the complete code path for the paper: auction-scenario
construction, prompt templates, six-checkpoint inference, constrained and
free-form readouts, uncertainty analyses, exact strategic-game enumeration,
normalized phase maps, robustness analyses, validation, packaging, and the
anonymous LaTeX manuscript. It also contains the saved model outputs needed to
rerun all downstream analyses without new LLM inference.

## Repository map

- `robustness_extensions/scripts/`: all runner and analysis source code;
- `robustness_extensions/server_outputs/raw/`: saved model generations;
- `preserved_outputs/qwen3_14b/`: preserved Qwen3-14B historical and synthetic generations;
- `preserved_outputs/granite_3_2_8b_instruct/`: preserved Granite historical and synthetic generations;
- `preserved_outputs/synthetic_existing4/`: preserved synthetic generations for the original four checkpoints;
- `docs/REPRODUCIBILITY_MANIFEST.json`: single machine-readable protocol, environment, revision, and provenance record;
- `offline_replay/`: analysis-only replay code and generated workspace;
- `robustness_extensions/server_outputs/data/`: instructions for locally
  reconstructed scenario inputs;
- `robustness_extensions/server_outputs/tables/`: machine-readable results;
- `robustness_extensions/server_outputs/figures/`: generated figures;
- `paper/`: anonymous NeurIPS EconML workshop LaTeX source;
- `docs/REPRODUCIBILITY.md`: recovered environment and model snapshots;
- `docs/RESULTS_PROVENANCE.md`: mapping from manuscript results to code and CSVs;
- `data/`: source-data acquisition, checksum, and licensing caveat;
- `environment/`: recovered server environment.

No model weights or raw auction CSV are committed. The canonical replay retains
the checksum-verified transformed scenario and split artifacts needed to audit
the reported scoring. Because the source mirror gives no clear redistribution
license, these artifacts are provided only for verification and are not
relicensed; see `LICENSES.md` and `data/README.md`.

## Reproduce analyses without model inference

Create an analysis environment and run the canonical six-checkpoint replay:

```bash
python -m pip install -r offline_replay/requirements.txt
export REPRO_PYTHON_BIN=/path/to/environment/bin/python
bash reproduce_all_from_preserved.sh
```

This verifies and reads the checked-in JSONL and regenerates all core,
free-form, stochastic-bootstrap, normalized-game, beta/gamma, menu/frontier,
auction-CDF, runtime/cost, zero-compute, and synthetic-IPV outputs. It never
loads a Hugging Face model. See `offline_replay/README.md` and
`docs/RESULTS_PROVENANCE.md`.

Validate the complete release and its integrity manifest with:

```bash
python validate_repository.py
shasum -a 256 -c MANIFEST.sha256
```

Compile the frozen manuscript presentation layer with:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Reproduce from the source dataset and checkpoints

Acquire and verify the dataset as described in `data/README.md`, placing it at
`online-auctions-dataset/auction.csv`. On a CUDA server, run:

```bash
export ROBUSTNESS_PYTHON_BIN=/path/to/python
export ROBUSTNESS_HF_CACHE=/path/to/huggingface/cache
# Optional and required only for price-calibrated direct games:
export ROBUSTNESS_GPU_HOUR_USD=0.74
bash robustness_extensions/scripts/run_4090_full.sh
```

The JSONL runner checkpoints every completed scenario/budget/seed cell and can
resume safely. The original four model commits are pinned in
`robustness_extensions/scripts/config.json` and
documented in `docs/REPRODUCIBILITY.md`. Tokenizers use the same pinned
repository revision unless a separate `tokenizer_revision` is configured.
Qwen3-14B and Granite are pinned separately under
`new_models_full_server_package/config/`.

The direct prompted games are retained for reproducibility but are not used in
the paper's reported strategic claims.

## Qwen3-14B and Granite-8B

The paper includes complete Qwen3-14B and Granite-8B historical and synthetic-IPV
panels. Their requested and resolved model/tokenizer revisions were recorded
contemporaneously. `qwen3_14b_addition/` and
`new_models_full_server_package/` retain server runners; the canonical replay
uses only the preserved outputs and does not rerun inference. Granite's
`thinking=True` template is disclosed as a cross-family protocol sensitivity.

## Seeds

- master data/scenario seed: 20260819;
- deterministic option-order seeds: 11, 23, 37;
- stochastic generation seeds: 1, 7, 21, 42, 99;
- fixed stochastic-study option order: 314159;
- direct-game experiment seeds: 101, 202, 303;
- primary stochastic/hybrid bootstrap seed: 20260819;
- free-form and auction-CDF robustness seed: 20260822.

Per-row generation seeds and candidate/order seeds are deterministically derived
with SHA-256 and stored in the output records.

## Important archival limitations

The original-four model commits were the only snapshots found in the server
cache during inspection and are recorded in `docs/HF_CACHE_AUDIT.json`.
Historical JSON metadata did not independently log them, so exact historical
identification assumes the cache was unchanged. The
exact historical versions of every transitive Python dependency and the value
of `PYTHONHASHSEED` were not saved. The repository states these gaps instead of
inventing them.

The original project software is released under MIT. The dataset mirror reports
an unknown license, so the raw source is not redistributed; see `LICENSES.md`
and `data/README.md`. Preserved processed artifacts are included only to audit
and replay the reported experiment and are identified by checksum in the manifest.
