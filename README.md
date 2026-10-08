# CPU-completed manuscript revision and reproducibility package

8 October 2026. Latest stage: saved-decision audit and human-study preparation. This package integrates the original preserved inference archive, the focused manuscript revision, the accuracy/utility comparison, and the completed CPU uncertainty and alternative-mapping extensions. No new inference was performed.

## Current result and scope

All six accuracy-minus-utility private threshold differences have positive exploratory simultaneous 95% bands. The DeepSeek 128/128-versus-768/768 selection-score contrast persists in all 10,000 paired family/seed draws. Profit-based quality and client discrimination remain material modeling choices; some profit frontiers do not support a high/high equilibrium at positive cost in many draws. These are conditional model results, not independent confirmation or evidence of deployed market inefficiency.

The arithmetic task, calculator superiority, budget ceiling, symmetric primitives and unestimated client-choice/value mappings remain limitations. Richer tasks and matched above-ceiling frontiers require new inference. Client choice and value require external evidence. Target-venue formatting, author/anonymity metadata and required supplementary/checklist details remain pending. The current anonymous workshop format is retained until a venue is selected.

See `decision_audit/results_report.md` for the calculator audit and `decision_audit/human_study/protocol.md` for the prepared pilot. No human responses have been collected. The local HTML survey passed static flow checks; live browser rendering was blocked by URL policy.

## Contents

- `decision_audit/`: compact input data, new analysis, human-study materials and submission readiness checks.
- `paper/`: latest editable manuscript; main text is nine pages, with references/appendices/checklist in the full PDF. Original numbered equations and original supporting assets are preserved; new comparison tables and analysis explanation are added.
- `submitted_paper/`: original submitted manuscript snapshot.
- `cpu_revision/results_report.md`: numerical findings and remaining claim boundaries.
- `cpu_revision/service_model_specification.md`: service participants, payment accounting, profit-based benefit, noisy quality-difference selection and identification limits.
- `cpu_revision/results/`: 10,000-draw uncertainty outputs, alternative mappings, retained bootstrap thresholds, seed/input hashes and validation.
- `cpu_revision/reference_evidence/`: previous focused-revision CPU results.
- `cpu_revision/replay_reference_tables/`: 71 preserved CSV tables for replay comparison.
- `cpu_revision/analyze_legacy_revision.py`: regenerates previous CPU evidence, including family-clustered endpoints and complete equilibrium regimes.
- `offline_replay/`, `preserved_outputs/`, `robustness_extensions/`, `new_models_full_server_package/`: original scripts and preserved data/generations needed to replay the reported experiments without loading model weights.
- `docs/REPRODUCIBILITY_MANIFEST.json`: updated original provenance manifest plus revision stage information.
- `MANIFEST.sha256`: current static-file checksums. Historical release metadata is retained under `docs/SUBMITTED_*`.

The original archive's licenses and transformed-data provenance limits remain in `LICENSES.md` and the provenance manifest. The public GitHub repository contains this revision. The anonymous mirror must be refreshed separately after publication.

## Reproduce all CPU work

Requires Python 3.13 and the pinned CPU analysis dependencies. With an existing compatible environment, run this single sequential command from the extracted package directory:

```sh
bash reproduce_cpu_revision.sh
cpu_revision_exit=$?
echo "Exit status: $cpu_revision_exit"
test "$cpu_revision_exit" -eq 0
```

To prepare an environment first:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r offline_replay/requirements.txt
REPRO_PYTHON_BIN="$PWD/.venv/bin/python" bash reproduce_cpu_revision.sh
cpu_revision_exit=$?
echo "Exit status: $cpu_revision_exit"
test "$cpu_revision_exit" -eq 0
```

Dependency installation may require network access; the replay and analysis themselves do not call model APIs, download weights or use a GPU. The wrapper verifies static-file checksums, executes the original full replay, regenerates previous and new CPU analysis, exports the revised tables and checks outputs. Generated `offline_replay/outputs/` and `cpu_revision/legacy_results/` are omitted from the downloadable archive and rebuilt by this command.

For an already completed full replay, `bash reproduce_cpu_revision.sh --use-existing-replay` reruns revision analyses and checks the existing 71 replay tables. This mode requires those generated tables to exist; it is not the first-run command.

To compile the manuscript after analysis:

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

A LaTeX installation is only needed for compilation, not numerical replay. Poppler is only needed for PDF rendering. Source hashes should be regenerated after intentional source edits; the wrapper deliberately stops when static inputs have changed.

## Validation performed for this release

The original complete replay exited successfully. All 71 archived CSV tables are byte-identical after replay. Four previous CPU evidence tables match the rerun. Independent LP/control equilibrium checks and explicit service-payment cancellation pass. The 10,000 new paired draws are retained, including nonpositive high/high support bounds; no invalid-support draw is silently discarded from absolute-difference inference. Ratio intervals with positive-support conditioning are labelled separately and are not used for the primary uncertainty conclusion.

The integrated wrapper's existing-replay path passed with the saved-decision stage included. The original full replay and its component checks also passed. Repository synchronization validation is recorded in `docs/PUBLICATION_VALIDATION.json`. See `docs/SAVED_DECISION_AUDIT_VALIDATION.json` for the latest compilation, inspection and human-study status; `docs/CPU_REVISION_VALIDATION.json` records the preceding CPU stage.
