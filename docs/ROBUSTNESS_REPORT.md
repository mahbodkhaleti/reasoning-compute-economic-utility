# Robustness extension status

Status date: 2026-08-19. “Addressed” means a result was computed and audited.
“Partially addressed” means executable code and an analysis plan exist, but the
required new model inference has not yet been run. No pending result is used as
paper evidence.

## Addressed

### Scenario-sampling uncertainty

- `scripts/bootstrap_uncertainty.py` performs 1,000 paired scenario-level
  bootstrap draws for each of the four models.
- A sampled scenario contributes its complete seven-budget outcome vector.
- The script recomputes every `q(c)` and all 2,400 strategic-game grid points for
  every draw.
- Pointwise `q(c)` estimates, bootstrap standard deviations, regret intervals,
  and percentile 95% intervals are in `uncertainty/`.
- This captures finite-scenario sampling uncertainty only.

### Strategic-game uncertainty and welfare loss

- Each bootstrap frontier is propagated through Nash and planner computation.
- PoRC remains a compute ratio, not welfare loss.
- Absolute loss is `W_OPT - W_NE`; relative loss divides by `|W_OPT|` only when
  `|W_OPT| > 0.01` normalized welfare units.
- Plug-in arms-race shares and scenario-bootstrap intervals are:

| Model | Arms-race share | 95% bootstrap interval | Conditional mean PoRC interval |
|---|---:|---:|---:|
| Phi-mini-reasoning | 71.79% | [47.49%, 82.30%] | [1.779, 4.206] |
| Qwen3-4B | 72.92% | [47.83%, 83.46%] | [4.944, 6.000] |
| DeepSeek-R1-Distill-Llama-8B | 89.25% | [75.88%, 93.79%] | [1.991, 2.014] |
| Qwen3-8B | 72.79% | [47.62%, 84.29%] | [4.683, 5.474] |

These intervals are wide and are reported as such. The `V`, `lambda`, `gamma`,
and `beta` grids remain economic sensitivity designs, not statistical samples.

## Partially addressed: code complete, 4090 inference pending

### Benchmark leakage and missing baselines

- Independent candidate sets are generated without conditioning on the optimal
  bid.
- Option order is randomized and stored for every observation.
- Calculator, uniform-random, median-position, highest-win-probability, and
  held-out-oracle policies are implemented.
- `no_think` and unconstrained model baselines are implemented.
- The intended output files exist with `pending_gpu_run` markers and will be
  overwritten by `analyze_extensions.py` after inference.

### Generation uncertainty

- The stochastic robustness run uses budgets 128, 384, and 768; seeds 1, 7, 21,
  42, and 99; `temperature=0.6`; and `top_p=0.95`.
- Option order is held fixed at seed 314159 so generation variability is not
  confounded with presentation order.
- Greedy runs are separate; their random seeds have no inferential meaning.
- Clustered confidence intervals and bid variance are implemented in
  `analyze_extensions.py`.
- The updated run additionally records answer-token probability and verbal
  self-confidence. Pending analyses include ECE, Brier score, NLL, AUROC,
  reliability diagrams, selective abstention utility, risk--coverage curves,
  confidence-weighted convergent sampling, and maxwise split-conformal
  abstention over budget checkpoints.
- A hierarchical scenario/generation bootstrap will propagate stochastic
  frontier uncertainty through a restricted three-budget strategic game.
- These additions are inspired by confidence-aware scaling and statistical
  stopping work, but do not reproduce training-time confidence dilution or a
  renewal process over uncertainty keywords.

### Direct strategic behavior

- `run_direct_game.py` makes two same-checkpoint agents choose compute before
  bidding in repeated auctions and compares adaptive behavior with fixed-low and
  fixed-high policies.
- Choice validity, budget dynamics, private payoff, compute cost, allocative
  loss, welfare, and ex-post policy comparisons are recorded.
- This concern remains empirically unresolved until the direct games run.

### Cost and social-value calibration

- Wall time, NVML power, energy, peak memory, GPU rental cost, and electricity
  cost are measured per generation.
- The user must supply the actual 4090 rental price; electricity price is
  optional. Thus monetary calibration is not yet a result.
- Auction values and held-out product-price distributions are exported for a
  transparent social-value scale, but alternative normative valuations still
  require judgment and sensitivity analysis.

### Multi-product evaluation and statistical comparisons

- Palm Pilot M515, Xbox game console, and Cartier wristwatch are included with a
  deterministic train/test split.
- Model-size-by-budget summaries, scenario-clustered log-budget regressions, and
  paired endpoint bootstrap comparisons are implemented.
- These remain pending until the four-model inference is complete.

## Still unresolved after the planned run

- The direct game uses a stylized repeated first-price environment and only
  same-checkpoint pairs; it does not identify behavior in deployed markets.
- Rental and electricity prices cover private resource cost but not every social
  externality, latency cost, carbon intensity, or opportunity cost.
- Three product categories and four checkpoints remain limited external
  validity.
- Reasoning tokens are not a hardware-invariant compute measure.
- Bootstrap and seed repetitions do not capture model misspecification,
  contest-function misspecification, distribution shift, or all deployment
  uncertainty.
- The economic mapping from auction utility ratio to contest success remains an
  assumption unless replaced by sufficiently powered direct head-to-head data.

## Reproduction

Run `scripts/run_local_uncertainty.sh` for the completed bootstrap. On the 4090,
set the actual `ROBUSTNESS_GPU_HOUR_USD` and execute
`scripts/run_4090_full.sh`. All inference is checkpointed. Send back
`scripts/robustness_4090_results.zip` for the final empirical paper update.
