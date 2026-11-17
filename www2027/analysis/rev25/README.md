# v25 revision analyses (CPU only, saved outputs)

All scripts reuse the paper's code path through `analysis/cpu_revisions/cr_common.py`:

- the 2% matcher;
- the text reader;
- hybrid written values;
- gap categories;
- the family bootstrap.

Outputs are in `results/rev25/`. No model was run again.

| Script | Item | Output |
|---|---|---|
| `r1_firstprice.py` | 1. First-price equilibrium win curve: rescore every decision and the calculator, with welfare | `r1_firstprice.json` |
| `r2_code_check.py`, `r2b_reader_only_split.py` | 2. Arithmetic checked by code on all 15,552 traces; lower and upper bounds on Arith. | `r2_code_check.json`, `r2b_reader_only_split.json` |
| `r3_stated_answer.py` | 3. Rule-based parser for the answer the trace states; split recomputed with it | `r3_stated_answer.json` |
| `r4_audit_oos.py` | 4. Audit out of sample (synthetic → eBay, eBay → synthetic, nested leave-one-product-out and family 6-fold); call-rate curves | `r4_audit_oos.json` |
| `r5_passover_regression.py` | 5. One logistic regression for passing over, with a scenario random-effect check | `r5_passover_regression.json` |
| `r6_r8_robustness.py` | 6. Landscape bootstrap. 7. Hierarchical bootstrap. 8. Break-even against the distribution of expected profit | `r6_r8_robustness.json` |
| `fig_main_v25.py` | Figure 1. Panel (c) is now the gap recovered against the call rate | `paper/figures/main_figure_v25.pdf` |

## Main results

### 1. First-price equilibrium scoring

The value distribution is lognormal per product, fitted by maximum likelihood. The fit treats each closing price as the second-highest of n bidders' values, with the median n between 7 and 9.

| Result | Value |
|---|---|
| Calculator q | 0.970 |
| Calculator matches the FP-optimal bid | 58% of scenarios |
| Gap recovered at 768 tokens | 40–85% (empirical curve: 42–85%) |
| Table 2 shares | move by at most 10 points |
| Moving answers to the calculator's, per 1,000 decisions | seller revenue −$13k to −$55k; bidder surplus +$12k to +$28k |
| Wrong answers that overbid | 47–66% |

### 2. Code-only arithmetic check

- **Agreement between two code methods:** where both find a value, the reader's value equals the true value in 86–95% of options.
- **Upper bound on arithmetic:** counting every misread value as miscomputed, Arith. is at most 17% of the gap for five models. For Granite-8B it is 45%, against 37% Passed. So Granite's split between Arith. and Passed is uncertain.
- **Human labels:** none exist.
  - TODO (author): hand-label 100–200 random traces with the sheet in `human_annotation/annotator_package/`.
  - Score them with `human_annotation/for_authors_only/score_human_labels.py`.
  - Report precision and recall in the camera-ready version.

### 3. Phi-mini's readout

| Model | Stated answer agrees with the letter |
|---|---|
| Qwen3 models (768 tokens) | 96–98% |
| Granite-8B (finished traces) | 93% |
| Phi-mini | 57% |

- Where Phi-mini's trace states a choice, that choice is the best option 94% of the time. Its letter answer is the best option 55% of the time.
- 156 of Phi-mini's passed-over letter answers come from traces that state a choice. In 139 of them, the stated choice is the best option.
- Using the stated answer where there is one raises Phi-mini's gap recovered from 42% to 64% [56, 71].

### 4. Audit, out of sample

| Setup | Gap recovered |
|---|---|
| Settings chosen on synthetic, tested on eBay | 84–100% |
| Settings chosen on eBay, tested on synthetic | 81–100% |
| Nested, leave one product out | 84–100% |
| Nested, family 6-fold | 84–100% |

At 20% of decisions sent to the calculator, the mean gap recovered is:

| Trigger | Gap recovered |
|---|---|
| Audit | 89% |
| Low letter probability | 81% |
| Unfinished list | 78% |
| Random | 74% |

### 5. Regression for passing over

The sample is 3,519 traces in which the best option is written and ranked first; 1,074 of them pass over it.

| Predictor | Odds ratio |
|---|---|
| Best option is the highest bid | 0.36 [0.28, 0.46] |
| Relative profit gap | 0.36 [0.13, 0.99] |
| Finished list | 0.38 [0.29, 0.51] |
| K = 7 | 1.89 [1.48, 2.41] |
| Value | 1.08 (not significant) |
| Position, pooled | 1.93 |

- **Position by model:** Phi-mini 17.8, Qwen3-8B 4.4, Qwen3-14B 2.2, Qwen3-4B 1.7 (not significant), DeepSeek-8B 1.3 (not significant), Granite-8B 0.15.
- **Mixed model with a scenario random intercept:** gives the same odds ratios.

### 6. Landscape bootstrap

The construction auctions, the held-out auctions, or both were resampled 1,000 times.

- The calculator stays ahead of every model in at least 96% of draws: 96.3% for DeepSeek-8B with both resampled, and 99.8% or more for the others.
- DeepSeek-8B stays best and Phi-mini worst in 100% of draws.
- The full ranking stays the same in 56% of draws, because Qwen3-4B and Qwen3-14B tie at .916.

### 7. Hierarchical bootstrap (products, then families)

Intervals are much wider: Phi-mini's gap recovered is 42% [10, 59] and Granite-8B's 65% [35, 79]. The upper bound stays below 100% for every model.

### 8. Break-even

| Quantity | Value |
|---|---|
| Best expected profit per decision: min / median / max | $1.75 / $48 / $1,521 |
| Minimum relative to the 768-token B/e | 39–130 times |
| Median relative to the 768-token B/e | 1,076–3,608 times |
| Mean gain from 768 tokens relative to their cost | 4,200–13,900 times |
