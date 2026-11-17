# v9 numbers and their sources

New or changed in v9 (round-5 judge fixes). Every number below was printed from the listed output file.

| Where in paper | Numbers | Script → output |
|---|---|---|
| Table 1, Answer column | 85 [78,91], 42 [34,50], 72 [66,78], 66 [60,72], 72 [66,79], 65 [57,73] | analysis/ci_gap.py → out/gap_ci.json (unchanged from v8) |
| Table 1, ≥K−1 column | 96, 93, 84, 77, 84, 85 | t8b_verifier_variants.py → t8b_verifier_variants.json (gap_km1, budget 768) |
| Table 1, Any eval. column | 100, 98, 96, 92, 98, 81 | t8b_verifier_variants.json (gap_always, 768); matches t1 tol 0.02 |
| Table 1, Uncon. column; text "72–96% for five, 49% Phi" | 96, 49, 91, 87, 89, 72 | t8_fixes.py → t8_fixes.json F1.*.natural_all.gap_readout |
| Table 1, Dollar | 57, 25, 50, 51, 47, 75 | t8_fixes.json F4.*.all.dollar |
| Text "Cartier gap $113–$186, elsewhere $2–$15; weakest for five of six" | dollar_gap_at_0 by product | t8_fixes.json F4 |
| Table 2, Arith./Unfin./Passed with CIs | see table | t8_fixes.json F1.*.split_768 (residual 0 for all) |
| Text "4% of annotated traces" | 0.042 (n=143) | t8_fixes.json F1.annotated_arith_changes_ranking |
| Text conditional logit: Phi 12-fold, Qwen 3–35-fold (p≤0.01), Granite 0.06 (p<0.001), DeepSeek n.s. | OR 12.1, 3.45/34.9/13.5, 0.06, 0.67 (p=0.37) | t8_fixes.json F2.*.or_best_first_to_last, p_best |
| Text ignored picks: runner-up 29–68%, highest-prob 19–56%, Phi first-listed 38%, Granite last-listed 28% | | t8_fixes.json F2.*.ignored_pick |
| Text "complete trace scores 0.075 higher; calculator equally accurate" | 0.0751; 0.980 vs 0.980 | analysis/batch6_traces_v2.py → out/batch6_traces_v2.json selection |
| Text verifier at 256 lowers q by 0.002–0.046 | | t8b_verifier_variants.json (q_always − q_readout, 256) |
| Text K−1 never lowers q by more than 0.001; 77–96% at 768 | min −0.001 | t8b_verifier_variants.json |
| Figure 1c | readout / K−1 / any by budget | fig_main_v9.py from t8b_verifier_variants.json |
| Text voting 2×384 vs 1×768: 0.06–0.18 below | −0.063 to −0.176 | analysis/batch3_reviewer_checks.py → out/batch3_results.json voting_matched.*.2x384_vs_1x768.diff |

All other numbers are unchanged from v8; see v8_numbers.md.
