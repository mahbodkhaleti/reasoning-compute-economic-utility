# Repository map

Start with [`www2027/README.md`](../www2027/README.md), which maps every result in the paper to its script and output, and the [reproduction guide](REPRODUCE.md). [`release.json`](../release.json) lists the canonical paths.

## The paper

| Path | Purpose |
| --- | --- |
| `www2027/README.md` | Overview and a map from every result in the paper to its script and output |
| `www2027/paper/` | Paper source and PDF |
| `www2027/analysis/` | All analysis code for the paper (eBay and AuctionNet) |
| `www2027/results/` | Analysis outputs and saved decisions from the follow-up eBay runs and the AuctionNet runs |
| `www2027/gpu_followup/` | GPU runners for the follow-up eBay runs and the AuctionNet runs |
| `www2027/setup_paths.sh` | Points the scripts' fixed root at the folder and links the raw eBay traces |

## Preserved experiment archive

| Path | Purpose |
| --- | --- |
| `preserved_outputs/` | Saved outputs of the original eBay and synthetic runs for all six models |
| `offline_replay/` | CPU replay of the archive, pinned requirements and archive verifier |
| `robustness_extensions/` | Original runner and analysis code, server outputs and supplementary evidence |
| `cpu_revision/` | Uncertainty and mapping analyses, reference outputs and replay comparison tables |
| `decision_audit/` | Calculator, aggregation and validity analyses; a prepared human-study protocol |
| `synthetic_ipv_benchmark/` | Synthetic first-price task manifest and supporting analysis |
| `new_models_full_server_package/` | Run package for the added models, transformed scenarios and recorded results |
| `synthetic_existing4_server_package/` | Synthetic-run package for the original four models |
| `qwen3_14b_addition/` | Qwen3-14B run package and provenance |
| `environment/` | Recorded inference environment |
| `data/` | Source acquisition instructions and provenance; no raw source data |

The paper's analyses in `www2027/` read the original eBay traces from this archive; `www2027/setup_paths.sh` links them. Large files are kept because they authenticate or reproduce recorded results.

## Records

- [REPRODUCE.md](REPRODUCE.md): commands for checking and replaying results.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): hardware, software and checkpoint revisions for the original runs.
- `REPRODUCIBILITY_MANIFEST.json`: the structured version of that record.
