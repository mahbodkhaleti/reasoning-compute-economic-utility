# Reasoning budgets for numerical decision services

Reproducibility repository for **Evaluating Reasoning Budgets for LLM Decision Services: Auction Utility, Calculator Gaps, and Resource Accounting**, prepared for the ICAIF 2026 AI4F workshop.

[Read the current anonymous paper](ai4f_workshop/paper.pdf) · [LaTeX source](ai4f_workshop/main.tex) · [Download the compact CPU package](releases/ai4f_cpu_reference.zip) · [Reproduction guide](docs/REPRODUCE.md)

## Start here

The compact package contains the current paper, saved-number audit inputs and the prospective CPU reference. It avoids downloading the complete historical inference archive. Extract it and run:

```sh
bash reproduce_ai4f_workshop.sh
```

For a complete repository checkout:

```sh
make check
```

These checks use Python 3.10+ and Poppler. They verify saved values, source/PDF provenance, conditional algebra and CPU execution. They do not generate new model responses or demonstrate method performance.

| Task | Command | Requirements |
| --- | --- | --- |
| Check the current release | `make check` | Python 3.10+, Poppler; no GPU |
| Compile the current paper | `make paper` | LaTeX/acmart, latexmk, BibTeX, Poppler |
| Replay the complete experiment archive | `make reproduce` | Python 3.13 and pinned analysis dependencies; no GPU |
| Rebuild the compact download | `make package` | Python standard library |

The checked PDF is `ai4f_workshop/paper.pdf`. Local compilation writes to `ai4f_workshop/build/` and does not replace it. See the [reproduction guide](docs/REPRODUCE.md) for environment setup and the difference between verification and full replay.

## What the evidence supports

The empirical benchmark measures held-out bidder profit, regret, construction-calculator gaps, aggregation, output validity and recorded resources in controlled consumer-auction tasks with supplied probabilities and assigned reasoning budgets. It does not use financial deployment data. The calculator remains stronger in the reported aggregate profit comparisons.

Decision-sensitive extraction repair is a **prospective method**. The CPU reference and fixtures check its restricted source contract and conditional arithmetic; independent language-model/service evaluation, cost savings and financial transfer remain untested. The appendix service contest is an assumed allocation sensitivity model, not observed strategic LLM-agent behavior.

## Repository guide

| Area | Contents |
| --- | --- |
| [ai4f_workshop/](ai4f_workshop/README.md) | Current anonymous manuscript, compact evidence, CPU reference and audits |
| [docs/](docs/REPOSITORY_MAP.md) | Repository map, reproduction instructions, provenance and release history |
| [preserved_outputs/](preserved_outputs/) | Historical/synthetic model outputs preserved for replay |
| [offline_replay/](offline_replay/README.md) | Rebuild historical and synthetic analyses without inference |
| [cpu_revision/](cpu_revision/) and [decision_audit/](decision_audit/) | Saved-data analyses and supporting evidence |
| [paper/](paper/) and [submitted_paper/](submitted_paper/) | Earlier full-paper revision and original submission; not the current AI4F paper |
| [releases/](releases/) | Compact anonymous source/audit download |

The [complete directory map](docs/REPOSITORY_MAP.md) explains the retained model-run packages and older robustness evidence. Their paths remain stable so historical scripts and provenance hashes continue to work. [Release history](docs/RELEASES.md) separates the current paper from previous versions.

## Data, licenses and anonymity

Raw eBay auction-source data must be obtained separately under the source provider's terms; transformed scenarios and preserved outputs are included. See [data provenance](data/README.md) and [licenses](LICENSES.md). No model weights are distributed.

This repository contains the anonymous paper. The named submission copy is delivered separately. The anonymous mirror is a separate snapshot and requires its owner's refresh; a GitHub update does not automatically prove mirror synchronization.
