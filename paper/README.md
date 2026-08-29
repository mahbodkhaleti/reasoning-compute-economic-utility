# EconML 2026 workshop paper

Compile from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The submission uses the supplied NeurIPS 2026 double-blind workshop style. The
main text occupies nine pages; references, appendices, and the NeurIPS checklist
follow. The compiled manuscript is `main.pdf`.

The original-four outputs are under
`../robustness_extensions/server_outputs/`; Qwen3-14B, Granite-8B, and
synthetic-IPV outputs are under `../preserved_outputs/`. Regenerate the complete
six-model analysis without model inference from the repository root:

```bash
bash reproduce_all_from_preserved.sh
```

This verifies all six historical and synthetic panels and regenerates the core,
free-form, stochastic-bootstrap, held-out-CDF, normalized-game, zero-compute,
runtime/cost, and synthetic-IPV analyses. See `../offline_replay/README.md` and
`../docs/RESULTS_PROVENANCE.md` for the output mapping.

The dimensionless strategic phase analysis is regenerated separately:

```bash
python \
  ../robustness_extensions/scripts/analyze_normalized_game.py \
  --output-dir ../robustness_extensions/server_outputs
```

It reads only the saved stochastic constrained choices. It enumerates the
$(\kappa,\rho,\gamma)$ phase grid, derives exact private and planner transition
thresholds, creates both phase figures, and checks 432 legacy/normalized game
solutions for identical Nash and planner allocations. The validation result is
stored in `server_outputs/tables/normalization_validation.json`.

The zero-compute-augmented sensitivity is also fully offline:

```bash
python \
  ../robustness_extensions/scripts/analyze_zero_compute_augmented.py \
  --output-dir ../robustness_extensions/server_outputs \
  --samples 1000 --grid-points 181
```

It averages the stored deterministic reasoning-mode zero-budget observations
over the three option orders within scenario, joins them to the existing
five-seed stochastic 128/384/768 panel, and reruns exact enumeration and a
hybrid hierarchical bootstrap. It does not perform model inference and does not
overwrite the primary stochastic outputs.

The original-four historical metadata contain repository identifiers and most
generation settings but not resolved Hugging Face commits or complete package
versions. Appendix F reports the sole snapshots recovered in a post-hoc cache
audit and the associated assumption. Qwen3-14B and Granite-8B record requested
and resolved revisions contemporaneously.
