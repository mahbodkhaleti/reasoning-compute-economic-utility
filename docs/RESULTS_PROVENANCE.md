# Results and table provenance

The manuscript's LaTeX tables are checked in under `paper/tables/`. The files
below are the machine-readable sources or analysis outputs behind the reported tables.
The LaTeX formatting is currently a frozen presentation layer; there is not a
separate CSV-to-LaTeX generator, and this limitation is stated rather than
hidden.

| Manuscript result | Primary machine-readable source | Producing code |
|---|---|---|
| Deterministic endpoints | `fixed_budget_summary.csv`, `paired_budget_comparisons.csv` | `analyze_fixed.py`, `analyze_extensions.py` |
| Complete deterministic frontier | `fixed_budget_summary.csv` | `analyze_fixed.py` |
| Analytic/generation controls | `analytic_baselines.csv`, `baseline_comparison.csv` | `analyze_fixed.py`, `analyze_extensions.py` |
| Stochastic frontier | `seed_robustness_summary.csv`, `stochastic_frontier_bootstrap.csv` | `analyze_extensions.py`, `bootstrap_stochastic_game.py` |
| Free-form sensitivity | `freeform_sensitivity_summary.csv`, `freeform_sensitivity_by_budget.csv` | `analyze_freeform_sensitivity.py` |
| Normalized thresholds | `normalized_transition_boundaries.csv` | `analyze_normalized_game.py` |
| Zero-compute augmentation | `hybrid_zero_compute_transition_comparison.csv` | `analyze_zero_compute_augmented.py` |
| Beta/gamma sensitivity | `game_beta_sensitivity_submission.csv`, `game_gamma_sensitivity_submission.csv` | `analyze_submission_robustness.py` |
| Primary strategic game and uncertainty | `stochastic_game_uncertainty_summary.csv`, `stochastic_game_uncertainty_bootstrap.csv` | `bootstrap_stochastic_game.py` |
| Menu/frontier sensitivity | `compute_menu_robustness_submission.csv`, `compute_menu_robustness_bootstrap.csv` | `analyze_submission_robustness.py` |
| Auction-CDF uncertainty | `auction_cdf_bootstrap_summary.csv`, `auction_cdf_bootstrap_draws.csv` | `analyze_submission_robustness.py` |
| Runtime and monetary benchmark | `runtime_cost_recomputed.csv` | `recompute_runtime_cost.py` |
| Synthetic IPV endpoint and game | `synthetic_endpoint_bootstrap.csv`, `synthetic_gamma1_exact_thresholds.csv` | `analyze_synthetic.py`, `analyze_synthetic_game.py` |

For the original four models, historical paths are relative to
`robustness_extensions/server_outputs/tables/`. The six-model canonical replay
writes corresponding outputs under `offline_replay/outputs/historical_six/`.
Synthetic-IPV endpoints, frontiers, and phase maps come from
`analyze_synthetic.py` and `analyze_synthetic_game.py` and are written under
`offline_replay/outputs/synthetic_six/`. Model setup is sourced from
`docs/REPRODUCIBILITY_MANIFEST.json` rather than manually inferred from labels.

The historical `server_outputs/SHA256_MANIFEST.json` authenticates the package
returned by the server at packaging time. Later offline analyses added files
that are not present in that older manifest. Running an analysis can also
replace derived CSV/PDF outputs; raw JSONL inference outputs remain the source
of record.

The raw auction source is not redistributed because the mirror reports an
unknown license. `data/README.md` supplies its source checksum and deterministic
reconstruction path. The exact scenario and split artifacts used in the paper
are preserved for audit and offline scoring and pinned by SHA-256 in the
machine-readable manifest.
