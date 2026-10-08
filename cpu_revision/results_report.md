# Completed CPU extension results

8 October 2026. No new inference.

## Paired uncertainty

10,000 within-product value-family/seed draws; all four candidate variants and paired budget/metric vectors are retained. Generation seeds are sampled independently by model, jointly across budgets within each selected scenario occurrence. Products, prompts and scoring CDFs stay fixed.

| Model | Accuracy minus utility threshold | Simultaneous 95% band |
|---|---:|---|
| DeepSeek-8B | 0.000349269 | [0.000239351, 0.000459187] |
| Phi-mini | 0.00016576 | [2.59774e-05, 0.000305543] |
| Qwen3-4B | 0.000304752 | [0.000191597, 0.000417906] |
| Qwen3-8B | 0.00026853 | [0.000149736, 0.000387323] |
| Qwen3-14B | 0.000240945 | [0.00015072, 0.000331171] |
| Granite-8B | 0.000191175 | [6.42558e-05, 0.000318093] |

All six exploratory simultaneous bands exclude zero. The DeepSeek strict-dominance contrast at kappa=0.00015 survives all 10,000 draws. This is a bootstrap stability frequency, not a market probability or independent confirmation.

Twelve Phi utility draws have a nonpositive signed high/high bound. Absolute difference intervals retain those draws. Ratio intervals in the supplementary CSV condition on positive utility bounds and are labelled accordingly. They are not used for the primary uncertainty claim.

Only DeepSeek and Qwen3-14B have linear-minus-measured threshold intervals excluding zero. Four intervals include zero; Qwen3-4B and Granite show substantial upper-tail shape sensitivity. The small point changes do not justify a general claim that intermediate shape is unimportant.

## Alternative selection and client-value assumptions

Profit-based selection changes point private thresholds to 0.24–1.85 times the utility thresholds. High/high support is absent in 24.71% of Phi draws, 36.68% of Qwen3-4B draws, 9.42% of Qwen3-8B draws and 0.06% of Granite draws. DeepSeek and Qwen3-14B retain support in every draw.

A quality-difference logit at matched reference slope changes point thresholds by +1.40% to +4.28%. Holding its noise scale fixed makes selection invariant to common score shifts. Doubling or halving discrimination changes point thresholds to approximately 2.01–2.07 or 0.51–0.52 times the utility thresholds. The functional form and discrimination remain unestimated.

See service_model_specification.md for explicit service accounting, client-value units and assumptions. Neither model identifies real client choice, auction total surplus, service payment, richer task competence or unconstrained budget agreement.

## Remaining work

CPU uncertainty, explicit alternative mappings, manuscript integration and portable packaging are completed in this bundle. Venue-specific format, author/anonymity information and checklist requirements await the target venue. Richer tasks and matched above-ceiling evidence need new inference; market mappings need external evidence.
