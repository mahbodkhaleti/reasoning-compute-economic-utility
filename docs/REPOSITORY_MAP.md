# Repository map

Start with the [current AI4F paper](../ai4f_workshop/paper.pdf), its [source](../ai4f_workshop/main.tex), and [reproduction guide](REPRODUCE.md). The root [release.json](../release.json) records these canonical paths.

## Current paper and its compact checks

| Path | Purpose |
| --- | --- |
| `ai4f_workshop/main.tex`, `paper.pdf` | Current anonymous source and checked PDF |
| `ai4f_workshop/tables/`, `figures/` | Unchanged experimental result assets |
| `ai4f_workshop/audit_sources/` | Saved numeric inputs, hashes and prior source snapshots |
| `ai4f_workshop/cpu_reference/` | Restricted admission/repair core, contract and engineering fixtures |
| `ai4f_workshop/archive/v4/` | Byte-preserved source/PDF preceding the abstract availability sentence |
| `releases/ai4f_cpu_reference.zip` | Portable compact package; excludes the large historical inference archive |
| `scripts/` | Compact-package builder and release/link/bundle checks |

## Preserved experiment evidence

| Path | Purpose |
| --- | --- |
| `preserved_outputs/` | Six-checkpoint historical and synthetic inference archives |
| `offline_replay/` | CPU analysis replay, pinned requirements and archive verifier |
| `robustness_extensions/` | Original runner/analysis code, server outputs and supplementary evidence |
| `cpu_revision/` | Uncertainty/mapping analyses, reference outputs and replay comparison tables |
| `decision_audit/` | Calculator, aggregation and validity analyses; prepared human-study protocol |
| `synthetic_ipv_benchmark/` | Frozen synthetic task manifest and supporting analysis |
| `new_models_full_server_package/` | Full model-run package, transformed historical scenarios and recorded results |
| `synthetic_existing4_server_package/` | Historical four-model synthetic-run package |
| `qwen3_14b_addition/` | Retained Qwen3-14B addition package and provenance |
| `environment/` | Historical inference environment records |
| `data/` | Source acquisition instructions and provenance, not raw source data |

These are evidence and replay dependencies, not additional current manuscripts. Large files are retained because they authenticate or reproduce recorded results. A lightweight download is provided instead of removing them or rewriting history.

## Earlier manuscripts and records

- [paper/](../paper/) is the earlier complete revision; its `main.tex` is not the current AI4F source.
- [submitted_paper/](../submitted_paper/) is the original submitted snapshot.
- [Historical release instructions](HISTORICAL_RELEASE.md) preserve the preceding README exactly; use [REPRODUCE.md](REPRODUCE.md) for current commands.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) records historical hardware/software and retrospective checkpoint attribution.
- [RESULTS_PROVENANCE.md](RESULTS_PROVENANCE.md) explains older table sources; current compact checks live in [NUMBER_AUDIT.md](../ai4f_workshop/NUMBER_AUDIT.md).
