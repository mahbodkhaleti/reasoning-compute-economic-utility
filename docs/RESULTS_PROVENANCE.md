# Results and table provenance

The manuscript's LaTeX tables are checked in under `paper/tables/`. The files
below are the machine-readable sources or analysis outputs behind Tables 1--13.
The LaTeX formatting is currently a frozen presentation layer; there is not a
separate CSV-to-LaTeX generator, and this limitation is stated rather than
hidden.

| Paper table | Primary machine-readable source | Producing code |
|---|---|---|
| 1, deterministic endpoints | `fixed_budget_summary.csv`, `paired_budget_comparisons.csv` | `analyze_fixed.py`, `analyze_extensions.py` |
| 2, complete deterministic frontier | `fixed_budget_summary.csv` | `analyze_fixed.py` |
| 3, analytic/generation controls | `analytic_baselines.csv`, `baseline_comparison.csv` | `analyze_fixed.py`, `analyze_extensions.py` |
| 4, stochastic frontier | `seed_robustness_summary.csv`, `stochastic_frontier_bootstrap.csv` | `analyze_extensions.py`, `bootstrap_stochastic_game.py` |
| 5--6, free-form sensitivity | `freeform_sensitivity_summary.csv`, `freeform_sensitivity_by_budget.csv` | `analyze_freeform_sensitivity.py` |
| 7, normalized thresholds | `normalized_transition_boundaries.csv` | `analyze_normalized_game.py` |
| 8, zero-compute augmentation | `hybrid_zero_compute_transition_comparison.csv` | `analyze_zero_compute_augmented.py` |
| 9, beta/gamma sensitivity | `game_beta_sensitivity_submission.csv`, `game_gamma_sensitivity_submission.csv` | `analyze_submission_robustness.py` |
| 10--11, primary strategic game and uncertainty | `stochastic_game_uncertainty_summary.csv`, `stochastic_game_uncertainty_bootstrap.csv` | `bootstrap_stochastic_game.py` |
| 12, menu/frontier sensitivity | `compute_menu_robustness_submission.csv`, `compute_menu_robustness_bootstrap.csv` | `analyze_submission_robustness.py` |
| 13, auction-CDF uncertainty | `auction_cdf_bootstrap_summary.csv`, `auction_cdf_bootstrap_draws.csv` | `analyze_submission_robustness.py` |

All paths in the table are relative to
`robustness_extensions/server_outputs/tables/`. Table 14 records model setup
from `scripts/config.json` and the environment/revision record.

The historical `server_outputs/SHA256_MANIFEST.json` authenticates the package
returned by the server at packaging time. Later offline analyses added files
that are not present in that older manifest. Running an analysis can also
replace derived CSV/PDF outputs; raw JSONL inference outputs remain the source
of record.

Source and processed auction-derived data files are not redistributed because
the source mirror reports an unknown license. `data/README.md` supplies the
source checksum and historical hashes of the deterministic reconstruction.
Consequently the historical server-package manifest lists data files that are
intentionally absent from the public repository.
