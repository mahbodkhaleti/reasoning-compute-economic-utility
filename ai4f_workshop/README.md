# AI4F v4: CPU revision and conservative reference

Active anonymous manuscript: `main_ai4f_v4.tex`. The earlier AI4F v3, v2 and other workshop versions are preserved. All experimental results, result tables, intervals, original numbered equations and original bibliography bytes are unchanged. Five verified references are added separately in `method_references.bib`. No new GPU inference or scientific experiment was run.

## Build and checks

Requires a LaTeX installation with `acmart`, BibTeX, `latexmk`, `stfloats` and `balance`. Python 3.10+ suffices for the CPU reference and mathematical fixtures; they use only the standard library. Poppler (`pdftotext`, `pdfinfo`) is needed for the manuscript-number/layout audit.

```sh
make
make check
python3 cpu_reference/check_saved_protocol.py
python3 verify_numbers.py --no-write
```

The final PDF has 9 pages: main text and references on pages 1–7, Appendix A on pages 8–9, and Appendix B on page 9. Table 3 is cited and stays within Section 5 on page 4. Original propositions and equations retain their numbering.

The root CPU driver can run directly from a static checkout without LaTeX or a local compilation log. It checks the published PDF against its recorded clean-build audit; this does not perform a new compilation. After `make`, the same number audit checks the current local compilation log. `--no-write` preserves the published report and release hashes.

## Contents

- `CHANGELOG.md`: scope and changes.
- `REVIEW_REPORT.md`: final technical/scientific review and calibrated rating.
- `NUMBER_AUDIT.md`, `number_audit.json`, `verify_numbers.py`: saved-evidence number checks.
- `LITERATURE_AUDIT.md`: primary sources supporting five new references and distinctions from related work.
- `SUBMISSION_READINESS.md`: file checks, pending authorship and publication items, and scientific limits.
- `cpu_reference/`: executable conservative admission, decision arithmetic, repair priority, output-budget ledger, fixtures and public development-protocol compatibility check. See `CONTRACT.md` for exact assumptions.
- `check_method_bounds.py`, `method_math_checks.json`: conditional algebra checks; no extraction or model evaluation.
- `audit_sources/`: compact saved evidence, source hashes and prior manuscript snapshots.
- `changes_from_v3.patch`: exact final source diff.
- `package_validation.json`, `MANIFEST_SHA256.json`: fresh extracted-package checks and content hashes.

This reference supplies callback-driven CPU orchestration, but no model backend or evaluated service. It supports explicitly typed source fields and cannot certify unrestricted prose. This directory is the current anonymous AI4F reproducibility release, added alongside the preserved historical evidence. The authored submission copy is maintained separately so enabling PDFs on the anonymous mirror does not disclose its author block. No new inference or submission occurred. Run the root `reproduce_ai4f_workshop.sh` for this release; the original CPU-replay driver remains unchanged.
