# Reproduce and verify

## 1. Current AI4F audits: compact and CPU-only

Download [ai4f_cpu_reference.zip](../releases/ai4f_cpu_reference.zip), extract it, enter `ai4f_cpu_reference/`, and run:

```sh
bash reproduce_ai4f_workshop.sh
```

Python 3.10+ supplies the CPU core's standard-library dependencies. Install Poppler separately for `pdftotext` and `pdfinfo`. There are no model downloads, APIs or GPU calls in this command.

In a full repository checkout, use `make check`. It additionally checks canonical paths, local documentation links, bundle integrity, the top-level static manifest and preserved archive coverage.

The compact audit verifies recorded numbers and runs engineering/algebra fixtures. It does not reconstruct all historical bootstrap outputs or validate the proposed method on independent model responses. The static PDF check compares the shipped PDF with its clean-build record; it does not compile LaTeX.

## 2. Compile the current anonymous paper

With LaTeX/acmart, latexmk, BibTeX and the source's packages installed:

```sh
make paper
```

Compilation writes `ai4f_workshop/build/main.pdf`; the subsequent audit checks that newly generated PDF and log. The checked release at `ai4f_workshop/paper.pdf` remains unchanged. No error-free build is claimed without running this command or consulting the shipped build report.

The compact bundle uses `make -C ai4f_workshop` followed by `make -C ai4f_workshop audit`. Older `paper/main.tex` and archived v4 snapshots are separate historical sources.

## 3. Complete preserved-result replay

From the complete repository, prepare the analysis environment:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r offline_replay/requirements.txt
make reproduce PYTHON="$PWD/.venv/bin/python"
```

This delegates to the preserved `reproduce_cpu_revision.sh` pipeline: original six-checkpoint replay, saved-data extensions, calculator audit and human-study status checks. It uses saved outputs, not new inference. Installing dependencies may require the network. The raw auction source is acquired separately; transformed task artifacts are included with the provenance limits in [data/README.md](../data/README.md).

Full replay requires the large archive and more CPU time/storage than the compact audit. It regenerates analysis products and may replace tracked derived outputs. Use a clean checkout and keep the original static manifest for comparison. Do not mistake regenerated outputs for new scientific observations.

After a complete replay, the existing shortcut is:

```sh
REPRO_PYTHON_BIN="$PWD/.venv/bin/python" bash reproduce_cpu_revision.sh --use-existing-replay
```

That shortcut requires the generated replay tables; it is not a first-run command. Historical details remain in [HISTORICAL_RELEASE.md](HISTORICAL_RELEASE.md), [offline_replay/README.md](../offline_replay/README.md), and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## 4. Maintainer packaging

`make package` builds the compact ZIP deterministically from the workshop static manifest. It refuses to package changed assets with stale hashes. An intentional release change requires updating the workshop build/number reports and workshop manifest, building the ZIP, staging the files, then regenerating `MANIFEST.sha256` with `python3 update_manifest.py` and rerunning `make check`.

Local build products, replay caches, virtual environments and human responses are ignored. Existing shell entry points remain available for compatibility. `release.json` is the canonical paper/path index; [REPOSITORY_MAP.md](REPOSITORY_MAP.md) separates current assets from historical dependencies.
