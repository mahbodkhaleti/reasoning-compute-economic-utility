# Paper integration checklist (after results exist)

Do not update claims or numbers until `qwen3_14b_complete_validation.json`
reports `PASS`. Then use the generated files as follows.

| Paper component | Qwen3-14B source output |
|---|---|
| Main compute--utility results | `tables/model_frontiers.csv`, `tables/fixed_budget_summary.csv` |
| Paired compute trend | `tables/paired_budget_comparisons.csv`, `tables/budget_regression.csv` |
| Stochastic frontier uncertainty | `tables/stochastic_game_uncertainty_summary.csv` |
| Free-form sensitivity | `tables/freeform_sensitivity_summary.csv`, `tables/freeform_sensitivity_by_budget.csv` |
| Confidence appendix | `tables/confidence_calibration_summary.csv`, `tables/adaptive_sampling_summary.csv` |
| Product/order robustness | `tables/multi_product_summary.csv`, `tables/randomized_order_results.csv` |
| Beta/gamma comparative statics | `tables/game_beta_sensitivity_submission.csv`, `tables/game_gamma_sensitivity_submission.csv` |
| Six-budget menu sensitivity | `tables/compute_menu_robustness_submission.csv` |
| Auction-CDF uncertainty | `tables/auction_cdf_bootstrap_summary.csv` |
| Normalized strategic phase map | `tables/normalized_transition_boundaries.csv`, `figures/normalized_phase_*.pdf` |
| Zero-compute augmentation | `tables/hybrid_zero_compute_*.csv`, `figures/hybrid_zero_compute_*.pdf` |
| Reproducibility | `raw/*_run_manifest.json`, `raw/*_metadata.json`, `SHA256_MANIFEST.json` |

When integrating, preserve the paper's current interpretation: empirical
reasoning-compute--utility measurement is the primary contribution; the game
maps measured frontiers into conditional comparative statics and does not
identify real-world strategic overcomputation.
