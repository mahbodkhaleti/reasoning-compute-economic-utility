# Every new or changed number in paper v8, and the script that produced it

Scripts are in analysis/revision/ unless marked "(analysis/)". Outputs in the matching out/ folder.

| Where in paper | Number(s) | Script → output |
|---|---|---|
| Abstract; Results "The trace already holds..." | best-evaluated readout recovers 80–100% of the gap at 768 tokens across 1/2/5% tolerance | t1_parser_robustness.py → t1_parser_robustness.json (gap_recovered_best_eval, budget 768) |
| Results "The trace already holds..." | 67–89% at 512 (best-evaluated, 2%); 17–56% at 512 (model readout) | t1_parser_robustness.py → t1_parser_robustness.json (budget 512) |
| Results "The trace already holds..." | 0.963 (annotator labels), 0.959 (parser), 0.897 (model readout) on 150 annotated traces | t1_parser_robustness.py → t1_parser_robustness.json (annotated_150) |
| Table 1, Best eval. column | 100%; 98% [97, 99]; 96% [94, 98]; 92% [88, 95]; 98% [96, 99]; 81% [75, 87] | t1_parser_robustness.py (tol 0.02, budget 768) |
| Table 2, Unfinished 2% column | 59% [42, 74]; 24% [16, 34]; 84% [75, 93]; 85% [76, 92]; 90% [81, 100]; 48% [38, 58] | t1_parser_robustness.py (tol 0.02, budget 768) |
| Table 2, Unfinished 1–5% column; Results "The arithmetic is right..." | 46–65%, 18–28%, 71–87%, 73–87%, 74–95%, 43–53%; Qwen 71–95% | t1_parser_robustness.py (tol 0.01 and 0.05) |
| Results "The arithmetic is right..." | Phi-mini ignores 32% (best first) vs 74% (best last); Granite-8B 79% vs 18%; Qwen and DeepSeek 8–11%; odds ratio 0.31–0.57 for best = highest-probability bid | t2_ignored_logit.py → t2_ignored_logit.json |
| Results "Forced reasoning recovers..." | synthetic traces 34–53% complete at 768 vs 53–90% on eBay; expected-profit CV median 0.50 vs 0.30; synthetic advantage 0.088 (p<0.001) → 0.030 (p=0.38) | t4_synthetic_vs_ebay.py → t4_synthetic_vs_ebay.json |
| Results "Spending tokens selectively" | nested router 0.54–0.71 (no probe), 0.61–0.74 (128-token probe), +0.012–0.031 q over a random mix at 512-token spend; leave-one-product-out 0.31–0.58 | t3_nested_routing.py → t3_nested_routing.json |
| Methods "Models and protocol" | 9–35 s per 768-token decision; under one cent per decision | t7_costs.py → t7_costs.json |
| Methods "Scenarios and estimator" | 41–103 held-out auctions per product; median s.e. of p 0.047 | (analysis/) batch7_judge_r3.py → out/batch7_results.json (R5) |
| Table 1, B/e column; Results "Costs" | 1.8, 1.7, 1.3, 2.3, 4.5, 2.2 cents; below $0.35 at $50/M; $15–$36 loss vs calculator | t7_costs.py → t7_costs.json |

Numbers carried over unchanged from v7 keep their earlier sources: frontiers and Δq (analysis/www_cpu_analyses.py), readout gap CIs (analysis/ci_gap.py → out/gap_ci.json), completion, medians and natural q (analysis/batch6_traces_v2.py), free-text 25% (analysis/batch7_judge_r3.py R1), dollar and synthetic gaps (analysis/routing_and_extras.py, batch4_judge_r1.py), single-signal areas 0.31–0.59 (analysis/routing_ratio.py, batch4_judge_r1.py F4, batch6_traces_v2.py), parser validation and kappa (analysis/ parser validation in out/parser_validation_v2.json and out/annotator_agreement.json).

Removed from v7: the highest-probability column and the 1,024-token projection in Table 2; the calculator ±20% noise sentence; the heterogeneity paragraph; the oracle router area; the per-token runtime range; the seed; per-product construction sizes.
