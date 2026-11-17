# Details moved out of the paper (v26)

Each item below has a single sentence in the paper. All numbers come from the scripts named.

## Seeds and GPU

- **Sampled seeds.** Five sampled seeds were run at 128, 384 and 768 tokens. The seed standard deviation of q is at most 0.017 (`analysis/revision/t13_judge10.py`, E7).
- **GPU for the 4,096-token run.** The 4,096-token run used an RTX 6000 Ada. The paper's runs used an RTX 4090, and an L40S for Qwen3-14B. Library versions and pinned weights were the same.
- **Drift check.** Ten forced-768 decisions per model were rerun on the RTX 6000 Ada. They gave the original chosen option in 38 of 40 cases (`gpu_followup/`, `compare_drift.py`).

## Alternative scoring curves

Gap recovered at 768 forced tokens under each alternative curve is in `analysis/cpu_revisions/README.md` (A8). Across lognormal, order statistic and first-price equilibrium it ranges over 40–87%.

## First-price equilibrium robustness (`r9_fp_robustness.py`)

Gap recovered at 768 forced tokens, in %:

| Variant | DeepSeek-8B | Phi-mini | Qwen3-4B | Qwen3-8B | Qwen3-14B | Granite-8B |
|---|---|---|---|---|---|---|
| lognormal | 85 | 39 | 74 | 67 | 74 | 75 |
| gamma | 86 | 39 | 75 | 68 | 76 | 75 |
| weibull | 86 | 38 | 75 | 68 | 75 | 75 |
| increment | 85 | 39 | 74 | 67 | 74 | 74 |
| n_minus_1 | 84 | 36 | 74 | 67 | 74 | 77 |
| n_plus_1 | 86 | 43 | 74 | 67 | 74 | 71 |

- Recovery differs from the paper's FP eq. column by at most 4 points.
- Seller revenue change when answers move to the calculator: −$12k to −$56k per 1,000 decisions.
- Bidder surplus change: +$10k to +$29k per 1,000 decisions.
- Share of wrong answers that overbid: 42–71%.

## Tolerance checks

Table 2 shares at matching tolerances of 1% and 5% are in `analysis/cpu_revisions/README.md`. No share moves by more than 11 points.

## Passing-over regression (`r5_passover_regression.py`)

- **Sample:** 3,519 traces at 768 tokens that write the best option's value and rank it first. 1,074 of them pass it over.
- **Model:** logit with standard errors clustered by scenario. The reference model is Qwen3-8B.

| Term | Odds ratio [95% CI] |
|---|---|
| Intercept | 0.68 [0.47, 0.99] |
| C(product)[T.Palm] | 0.50 [0.37, 0.68] |
| C(product)[T.Xbox] | 0.48 [0.35, 0.65] |
| C(mod, Treatment('Qwen3-8B'))[T.DeepSeek-8B] | 0.49 [0.35, 0.68] |
| C(mod, Treatment('Qwen3-8B'))[T.Granite-8B] | 3.04 [2.12, 4.34] |
| C(mod, Treatment('Qwen3-8B'))[T.Phi-mini] | 5.92 [4.27, 8.21] |
| C(mod, Treatment('Qwen3-8B'))[T.Qwen3-14B] | 0.74 [0.58, 0.94] |
| C(mod, Treatment('Qwen3-8B'))[T.Qwen3-4B] | 1.03 [0.80, 1.32] |
| gap12 | 0.36 [0.13, 0.99] |
| pos | 1.93 [1.42, 2.61] |
| K7 | 1.89 [1.48, 2.41] |
| logv | 1.08 [0.77, 1.51] |
| best_is_high | 0.36 [0.28, 0.46] |
| finished | 0.38 [0.29, 0.51] |
| C(mod)[DeepSeek-8B]:pos | 1.33 [0.63, 2.78] |
| C(mod)[Granite-8B]:pos | 0.15 [0.08, 0.27] |
| C(mod)[Phi-mini]:pos | 17.79 [9.27, 34.13] |
| C(mod)[Qwen3-14B]:pos | 2.22 [1.04, 4.75] |
| C(mod)[Qwen3-4B]:pos | 1.69 [0.88, 3.26] |
| C(mod)[Qwen3-8B]:pos | 4.40 [2.06, 9.41] |
| C(mod)[DeepSeek-8B]:best_is_high | 0.29 [0.10, 0.80] |
| C(mod)[Granite-8B]:best_is_high | 0.51 [0.31, 0.83] |
| C(mod)[Phi-mini]:best_is_high | 0.55 [0.34, 0.89] |
| C(mod)[Qwen3-14B]:best_is_high | 0.04 [0.01, 0.29] |
| C(mod)[Qwen3-4B]:best_is_high | 0.31 [0.16, 0.61] |
| C(mod)[Qwen3-8B]:best_is_high | 0.03 [0.00, 0.19] |

A Bayesian mixed logit with a scenario random intercept gives the same odds ratios. The scenario random-effect standard deviation is 0.56.

## Hierarchical and landscape bootstraps (`r6_r8_robustness.py`)

See `README.md` in this folder.
