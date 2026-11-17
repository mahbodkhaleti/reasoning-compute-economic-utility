# Reproduce and verify

## 1. The paper's numbers (CPU only)

```sh
bash www2027/setup_paths.sh
pip install numpy pandas scipy statsmodels matplotlib
make check
```

`make check` runs `validate_repository.py` (file hashes against `MANIFEST.sha256`, preserved-output coverage, scenario counts, and a compile check of all analysis code), then rebuilds the AuctionNet summary and every result table of the paper from the saved decisions. No model is downloaded and no GPU is used.

To rebuild a specific result, find it in the table in [`www2027/README.md`](../www2027/README.md) and run the listed script. Scripts in `www2027/analysis/rev25/` and `www2027/analysis/followup/` import `r_common.py`, so run them from `/tmp/www2027/analysis/rev25/`.

## 2. Compile the paper

With LaTeX and the ACM `acmart` class installed:

```sh
make paper
```

## 3. Replay the preserved eBay archive

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r offline_replay/requirements.txt
make reproduce PYTHON="$PWD/.venv/bin/python"
```

This runs `reproduce_cpu_revision.sh`: the replay of the original runs, the saved-data extensions and the calculator audit. It uses saved outputs, not new inference. The raw auction source is obtained separately ([data/README.md](../data/README.md)). The replay needs more CPU time and storage than `make check` and may rewrite tracked derived outputs, so use a clean checkout.

After a complete replay, the shortcut is:

```sh
REPRO_PYTHON_BIN="$PWD/.venv/bin/python" bash reproduce_cpu_revision.sh --use-existing-replay
```

## 4. Rerunning the models

GPU runners for the follow-up eBay runs and the AuctionNet runs are in `www2027/gpu_followup/followup_gpu/`, with setup scripts for one rented GPU. The original eBay runs used the code in `robustness_extensions/`, `new_models_full_server_package/` and `qwen3_14b_addition/`.

## Maintainers

After changing tracked files, regenerate the manifest and validate:

```sh
git add -A
python3 update_manifest.py
python3 validate_repository.py
```
