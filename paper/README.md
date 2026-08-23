# EconML 2026 workshop paper

Compile from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The submission uses the supplied NeurIPS 2026 double-blind workshop style. The
main text occupies nine pages; references, appendices, and the NeurIPS checklist
follow. The compiled manuscript is `main.pdf`.

The current four-model stochastic results and robustness outputs are under
`../robustness_extensions/server_outputs/`. The submission-specific
offline analyses can be regenerated without model inference from the saved
JSONL/CSV outputs:

```bash
python \
  ../robustness_extensions/scripts/analyze_submission_robustness.py \
  --output-dir ../robustness_extensions/server_outputs \
  --samples 1000 --auction-samples 2000
```

This produces the beta and gamma sweeps, the deterministic six-budget
menu/frontier comparison, the held-out auction-CDF bootstrap, and the strategic
sensitivity figure used in the appendix. The six-budget comparison changes both
the action menu and the frontier's decoding regime, so it is intentionally
reported as a combined menu/frontier sensitivity rather than a pure menu test.

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

The historical run metadata contain the model repository identifiers and most
generation settings, but did not contemporaneously log exact Hugging Face
commits or exact Transformers/Accelerate versions. Appendix~D reports the sole
snapshots recovered in a post-hoc cache audit and the versions recovered from
the preserved environment, with an explicit cache-immutability caveat. The
runner now pins and records those fields prospectively.
