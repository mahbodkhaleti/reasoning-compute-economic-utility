# Saved-decision and calculator audit

8 October 2026. No new inference.

## Main finding

The construction-split calculator earns mean expected bidder profit of $212.66, with mean held-out regret $3.09. Every greedy 768-token model remains below this profit baseline under the six-model simultaneous exploratory bands. Additional tokens improve LLM performance relative to zero; they do not establish a practical advantage over the numerical policy.

| Model | LLM minus calculator, USD | Simultaneous 95% band | Displayed-optimal actions |
|---|---:|---|---:|
| DeepSeek-8B | -22.60 | [-38.80, -6.39] | 86.3% |
| Phi-mini | -33.67 | [-55.78, -11.57] | 42.1% |
| Qwen3-4B | -30.54 | [-48.97, -12.11] | 71.1% |
| Qwen3-8B | -30.37 | [-45.84, -14.91] | 67.7% |
| Qwen3-14B | -36.00 | [-54.15, -17.86] | 76.7% |
| Granite-8B | -15.09 | [-24.89, -5.28] | 53.5% |

## What errors can and cannot identify

Actions are classified by whether they maximize expected profit under the four-decimal probabilities shown and under held-out scoring. The four classes partition every record. They measure observable objective agreement, not internal arithmetic competence, hallucination or a causal reasoning mechanism. Held-out utility optimality permits equally profitable bids, unlike the original exact-label accuracy statistic.

The full-precision calculator is held-out utility-optimal in 155/216 scenarios. Thus an action can be optimal for the supplied distribution yet lose held-out profit. At 768 tokens, displayed-optimal/held-out-suboptimal choices comprise 9.4–22.8% of model records; both-suboptimal choices comprise 11.9–52.0%. The displayed precision changes the full-precision calculator's tie-broken action in one scenario; the original baseline is preserved. The display-matched calculator loses only $0.0113 in mean held-out profit, and all six model deficits remain negative under separately recomputed simultaneous bands against it.

The exact decomposition is: model held-out regret = calculator held-out regret + (calculator held-out profit − model held-out profit). The second term is signed: a model can sometimes beat the calculator under the scoring distribution despite being worse under the displayed objective. Such events occur in 2.2–8.0% of 768-token records; they do not overturn the aggregate disadvantage.

Product-specific endpoint means and regret concentration are supplementary descriptive diagnostics. They use the constructed tasks and do not identify real-market task frequencies. The largest mean dollar-profit model need not be the largest mean utility-ratio model because these objectives weight task scales differently.

## Uncertainty and limits

10,000 paired within-product value-family draws (seed 20261009) retain four candidate variants and all budgets, after averaging the three deterministic option orders within each scenario. The six 768-token dollar-profit gaps receive simultaneous maximum-standardized-deviation bands. This conditions on original prompts, products, construction and scoring CDFs. No independent confirmation, market generalization or stochastic-generation uncertainty is claimed for this greedy audit.

Calculator elapsed-time/cost zeros in archived analytic rows are placeholders, not measurements. Therefore this is a gross expected-profit comparison, not a measured net-payoff or deployment cost-effectiveness ranking.

## Human study status

The accompanying human_study directory contains proposed participant materials and an offline survey. Human responses have not been collected. Hypothetical benchmark-profile choice can test responses to the displayed quality and fee information; it cannot alone identify deployed-market selection, actual willingness to pay, or realized client surplus.
