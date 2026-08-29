# Canonical offline replay

This directory regenerates the paper's tables and figures from preserved model
outputs. It never imports Transformers or performs model inference.

From the repository root, install the analysis dependencies and run:

```bash
python3 -m pip install -r offline_replay/requirements.txt
bash reproduce_all_from_preserved.sh
```

The command verifies the raw historical and synthetic cell counts, stages copies
under `offline_replay/outputs/`, and reruns deterministic, stochastic,
free-form, auction-CDF, strategic-game, phase-map, and synthetic-IPV analyses.
The original preserved files are read-only inputs. Outputs are written beneath
`offline_replay/outputs/`. The exact endpoint-bootstrap presentation is written
separately for `reported_original_four/`, `reported_qwen3_14b/`, and
`reported_granite_3_2_8b/` (the joint Qwen3-14B/Granite RNG stream), matching the seeded analysis panels integrated into
the manuscript. The `historical_six/` directory supplies combined six-frontier
plots and games.

The historical `analyze_fixed.py` used Python's randomized `hash()` only to
seed descriptive per-cell frontier intervals. The replay copy replaces that
call with SHA-256-derived seeds. Main paired endpoint intervals use their
original explicit NumPy seed and estimand.
