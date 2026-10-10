# Release history

## 10 October 2026: current AI4F package and navigation cleanup

- Canonical current paper: [anonymous PDF](../ai4f_workshop/paper.pdf) and [source](../ai4f_workshop/main.tex).
- Added the repository availability sentence at the end of the abstract. Existing result text, tables, confidence intervals, equations and citations remain unchanged.
- Retained the previous anonymous v4 source/PDF byte-for-byte in [archive/v4/](../ai4f_workshop/archive/v4/). The named-author submission remains separate.
- Replaced the mixed root README with one current-paper entry point, a complete directory map, distinct verification/replay instructions and a machine-readable release index.
- Added root Make targets and a compact deterministic CPU/source download. LaTeX build products now go in ignored `build/` rather than replacing the checked release PDF.
- Historical data, raw model outputs, scientific analysis code and replay paths remain intact. No inference, new experiment or submission occurred. The anonymous mirror's refresh is a separate owner action.

## Earlier AI4F v4 companion

The preceding source/CPU companion was added at `a547466`; `9464434` fixed the static-checkout audit's dependency on an unshipped compiler log. [archive/README.md](../ai4f_workshop/archive/README.md) explains the preserved snapshot and source dependencies.

## 8 October 2026: complete CPU manuscript revision

The [earlier full-paper revision](../paper/) and [historical release instructions](HISTORICAL_RELEASE.md) preserve saved-decision, uncertainty, mapping and human-study preparation work. The human-study protocol is prepared; no collected responses or validation claim is implied.

## Original submission and inference provenance

The [submitted snapshot](../submitted_paper/), [model/environment provenance](REPRODUCIBILITY_MANIFEST.json), and [historical results provenance](RESULTS_PROVENANCE.md) are retained as evidence, not presented as the current AI4F paper.
