#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOUT="$ROOT/results/historical"
python "$ROOT/src/analyze_fixed.py" --config "$ROOT/config/experiment.json" --output-dir "$HOUT"
python "$ROOT/src/analyze_extensions.py" --output-dir "$HOUT" --export-dir "$HOUT" --bootstrap-samples 2000
python "$ROOT/src/analyze_freeform_sensitivity.py" --results-root "$HOUT" --bootstrap-samples 2000
python "$ROOT/src/bootstrap_stochastic_game.py" --output-dir "$HOUT" --samples 1000
python "$ROOT/src/analyze_normalized_game.py" --output-dir "$HOUT"
python "$ROOT/src/analyze_zero_compute_augmented.py" --output-dir "$HOUT" --samples 1000
python "$ROOT/src/analyze_submission_robustness.py" --output-dir "$HOUT" --samples 1000 --auction-samples 2000
python "$ROOT/src/analyze_confidence_uncertainty.py" --output-dir "$HOUT" --export-dir "$HOUT" --bootstrap-samples 1000
python "$ROOT/src/estimate_runtime.py" --output-dir "$HOUT" --models qwen3_14b,granite_3_2_8b_instruct
python "$ROOT/src/analyze_synthetic.py" --output-dir "$ROOT/results/synthetic" --paper-ready "$ROOT/paper_ready/synthetic"
python "$ROOT/src/analyze_synthetic_game.py" --output-dir "$ROOT/results/synthetic"
python "$ROOT/src/augment_synthetic_summary.py" --output-dir "$ROOT/results/synthetic" --paper-ready "$ROOT/paper_ready/synthetic"
cp "$ROOT/results/synthetic/tables/synthetic_gamma1_exact_thresholds.csv" "$ROOT/paper_ready/synthetic/"
cp "$ROOT/results/synthetic/tables/synthetic_gamma1_exact_thresholds.tex" "$ROOT/paper_ready/synthetic/"
cp "$ROOT/results/synthetic/tables/synthetic_normalized_transition_boundaries.csv" "$ROOT/paper_ready/synthetic/"
cp "$ROOT/results/synthetic/figures/synthetic_phase_gamma1.pdf" "$ROOT/paper_ready/synthetic/"
cp "$ROOT/results/synthetic/figures/synthetic_phase_gamma1.png" "$ROOT/paper_ready/synthetic/"
cp "$ROOT/results/synthetic/figures/synthetic_phase_all_gamma.pdf" "$ROOT/paper_ready/synthetic/"
cp "$ROOT/results/synthetic/figures/synthetic_phase_all_gamma.png" "$ROOT/paper_ready/synthetic/"
cp "$ROOT/paper_ready/synthetic/RESULTS_SUMMARY.md" "$ROOT/paper_ready/SYNTHETIC_NEW_MODELS_SUMMARY.md"
python "$ROOT/src/make_paper_outputs.py" --package-root "$ROOT"
python "$ROOT/src/analyze_qwen_scale.py" --package-root "$ROOT"
python "$ROOT/src/analyze_all_model_frontier.py" --package-root "$ROOT"
