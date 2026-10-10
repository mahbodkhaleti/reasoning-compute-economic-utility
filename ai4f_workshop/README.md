# Current anonymous AI4F paper and CPU companion

**Evaluating Reasoning Budgets for LLM Decision Services: Auction Utility, Calculator Gaps, and Resource Accounting**

[Checked PDF](paper.pdf) · [LaTeX source](main.tex) · [Compact download](../releases/ai4f_cpu_reference.zip) · [Reproduction guide](../docs/REPRODUCE.md)

The abstract now includes the anonymous repository availability link. The named submission copy is delivered separately. All experimental results, result tables, intervals, numbered equations and citation occurrences are preserved. The source/PDF preceding this editorial addition are retained in [archive/v4/](archive/v4/).

## Run the compact checks

From the full repository or compact bundle root:

```sh
bash reproduce_ai4f_workshop.sh
```

Python 3.10+ and Poppler are required. No GPU, model download or API is used. The tests check saved evidence and conditional CPU behavior, not independent model/service performance. The PDF check authenticates the recorded clean build and explicitly reports that no new compilation occurred.

## Compile and audit locally

From this directory:

```sh
make
make audit
```

Requires LaTeX/acmart, latexmk, BibTeX and Poppler. Outputs go to `build/main.pdf`; the reviewed `paper.pdf` remains unchanged. The audit's `--built` option checks the new build log and PDF. `--no-write` keeps the shipped number report intact.

## Evidence and boundaries

- [NUMBER_AUDIT.md](NUMBER_AUDIT.md), `number_audit.json`, `verify_numbers.py`: saved-number and preservation checks.
- [cpu_reference/CONTRACT.md](cpu_reference/CONTRACT.md): restricted admission, exact arithmetic, repair orchestration and fixtures.
- [PROSPECTIVE_EVALUATION.md](PROSPECTIVE_EVALUATION.md): independent evaluation still required.
- `audit_sources/`: compact saved CSVs, source hashes and prior snapshots.
- `tables/`, `figures/`, bibliography files: preserved result assets.
- `package_validation.json`, `release_validation.json`, `MANIFEST_SHA256.json`: checked build, release evidence and hashes.
- [archive/README.md](archive/README.md): earlier v4 source/PDF and their unchanged assets.

The calculator remains stronger in the reported simultaneous dollar-profit comparisons. The new extraction-repair policy has no independent model/service evaluation; its narrow parser can read every field admitted by the conservative profile. CPU fixtures do not establish financial transfer, cost savings or general semantic containment. No new inference or submission occurred. The anonymous mirror is refreshed separately and should not be described as synchronized merely because GitHub was updated.
