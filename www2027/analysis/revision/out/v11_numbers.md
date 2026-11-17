# v11 numbers and their sources

v11 answers the round-7 judge. Every number was printed from the listed output. t10_judge7.py's own B2/B3 (reader-only split) are superseded by t10b_hybrid_split.py.

| Where in paper | Numbers | Script → output |
|---|---|---|
| Table 1 all columns with CIs (answer, text reader, oracle, dollar) | | t10_judge7.py → t10_judge7.json B8 (reader = segment reader, ≥2 values, 600-char window, from extract8.py → segments_hist.csv.gz) |
| Table 1 "1,024 run" with CIs | 96 [93,98] … 72 [66,77] | t8_fixes.json F1 natural_all gap_readout(_ci) |
| Text: reader at 512 65–88% vs answer 17–56%; min diff −0.026; windows ±4 points; K−1 73–92% | | t10_judge7.json B1 |
| Text: synthetic reader Phi 47→86, Granite 65→73, others −3 to −14 points | | t10_judge7.json B1.synthetic (segments_syn.csv.gz) |
| Text: audit flags 18–57%, answer wrong in 58–89% of flags | | t10_judge7.json B6 |
| Methods: miscomputed values 14% Granite, ≤3% others; annotators 16%, ≤1% | | t10b_hybrid_split.json miscompute_rate; t10c_annotator_rates.py → t10c_annotator_rates.json |
| Table 2 shares and CIs; text 62–93%, 2–34%, 22% / ≤5% | | t10b_hybrid_split.py → t10b_hybrid_split.json forced768 |
| Text: interruption 45–89% vs 14–62%; logit p=0.06 | | t10b_hybrid_split.json interruption |
| Text: natural stops right ≥97.6%; Granite unforced split 55/27/14 | | t10b_hybrid_split.json natural_stop |
| Text: synthetic position Phi 17 [6,48], Granite 0.09 [0.02,0.39] | | t10b_hybrid_split.json synthetic_position |
| Text: near-ties 54–63% vs 22–27%; Phi 26 vs 24, Granite 37 vs 30; chance ≈20% | | t10_judge7.json B5 |
| Text: overbidding 81–91%, 12–20% of value; Phi 60%, Granite 51%; payment +$18–$83; win prob within 4 points; loss/value by product | | t10_judge7.json B7 |
| Text: voting 0.05–0.19 | | t9_judge6.json A7 |
| Text: 46–87% at cap, Granite 90% by 669 | | t9_judge6.json A4 |
| Table 2 medians, completion agreement | | t9_judge6.json A3 |
| Odds ratios eBay with CIs and events; budget-0 repeats 32–61%, 21–33% | | t9_judge6.json A5 |
| Figure 1b, 1c | | fig_main_v11.py from t9_judge6.json A3 and t10_fig_series.json |

Numbers not listed are unchanged from v8–v10 (see v8/v9/v10_numbers.md).

## v12 additions (round-8 judge)
All from t11_judge8.py → t11_judge8.json:
- C1: overbid vs chance (81–91% vs 55–61%; Phi 60/51, Granite 51/46); highest-bid picks 40–48% vs 18–19%; Phi first-listed 43%/18%, last 1%; Granite runner-up 47%/20%, first 3%; budget-0 repeats 28–62% vs 17–18%; wrong-answer win +8–14 points, payment $239–$339 → $459–$717, lost $94–$164.
- C2: highest-bid share of missed loss 63–78%; near-tie runner-up 25–38 cases, ~0 loss; choice logit signs and p ≤ 0.011.
- C3: K=5 incomplete share Qwen 26–63%.
- C4: audit recall 53–88%, value flagged 54–99%, false alarms 7–24%, flag→calculator policy 84–100% with 18–57% calls.
- C5: re-scoring: lognormal 44–85%, shaded 34–83%; overbid of wrong 47–65% / 73–89%.
- C6: 0.075 [0.062, 0.088].
- C7: held-out 41/103/45; DeepSeek oracle = calculator 98%; Granite unforced incomplete 4%.

## v13 additions (round-9 judge)
All from t12_judge9.py → t12_judge9.json:
- D1: unforced finished-list pass-over 50% Phi, 40% Granite, 6–9% others (n 458–612); forced finished by 600 tokens 51% / 43% vs 7–17%; Granite unforced 0.912 vs forced 0.897.
- D2: truncation share (not reached + incomplete) Qwen 74–79%, DeepSeek 47%, Phi/Granite 17%.
- D3: synthetic audit policy 82–100%, recall 51–91%, false alarms 11–26%.
- D4: missed-best share on Palm/Xbox 63–100%.
- D5: eBay second-price rescoring: overbids gain $39–$121 vs calculator choice; budget-zero q 0.91–0.98 (five models) vs calculator 0.89.
- D6: DeepSeek oracle differs in 16 decisions, q 0.976 vs 0.987.
- D7: 512-token reader intervals (±6 points); natural-stop right rates.
Prompt wording: robustness_extensions/scripts/common.py auction_prompt (repository).

## v14 additions (round-10 judge)
All from t13_judge10.py → t13_judge10.json:
- E1 order-statistic curve: recovery 43–87%; overbid of wrong vs new optimum 38–57%.
- E2 per 1,000 decisions: pay-your-bid empirical revenue +$18k–$83k, surplus −$15k–$36k; order-statistic +$16k–$62k, −$5k–$14k; eBay rule revenue +$5k–$26k, surplus +$8k–$21k (five models).
- E3 reader rule chosen on Palm (600 chars, K−1); Cartier+Xbox reader 71–91% vs answer 29–80%; audit 73–93%, calls 13–50%.
- E4 alternative order: arithmetic Granite 22%, others 0–1%.
- E5 budget-0 repeats: same order 28–62%, other order 17–46%, bid-rank baseline 17–38%; Phi 45/17/17.
- E6 within-trace: right at 512 29–54%; finished by 768: DeepSeek/Qwen 84–89%, Phi 52%, Granite 46%.
- E7 seeds: s.d. 0.001–0.017; 5-vote +0.005–0.040; energy 2.3–10.5 kJ at 768 vs 0.03–0.26 kJ at 0.

## v15 additions (round-11 judge)
All from t14_judge11.py → t14_judge11.json:
- F1 first-price equilibrium curve: recovery 40–85%; overbid of wrong 47–66%.
- F2 K split; prefix analysis with five options: DeepSeek/Qwen right after finishing 88–91%.
- F3 right-despite-trace ≤3.7% of decisions.
- F4 audit vs triggers at matched rate (Phi 97 vs 87 low-prob, 82 random, 56 unfinished).
- F5 dollar-weighted split (Qwen3-8B unreached 48%).
- Difficulty: t10_judge7.json B5 coef_gap12 (p ≤ 0.01 for five models, Phi p = 0.10). Bonferroni note: t9_judge6.json A5 p_best (Qwen3-4B p = 0.010).
