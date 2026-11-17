# AuctionNet P0 results

Scripts: analysis/auctionnet/an_p0_analysis.py, an_split.py, an_regression_audit.py. Outputs: results/AN/*.json.
Calculator q = 0.976, random = 0.776. 648 decisions per model and setting.

| Model | q(0) | q(768) | Gap 768 | Gap unforced 1,024 | Unforced: mean tokens / hit cap | Dollar-pooled 768 | Finished 768 | Split 768: NR / Ar / Cut / Pass | Highest bid % of wrong (chance) | Audit (eBay setting): recovery / calls |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-8B | 0.726 | 0.956 | 92 [89, 95] | 98 [96, 99] | 1014 / 87% | 89 | 88% | 8 / 0 / 27 / 65 | 21 (17) | 99 / 15% |
| Phi-mini | 0.778 | 0.825 | 24 [15, 32] | 43 [35, 51] | 1022 / 97% | 26 | 93% | 3 / 1 / 6 / 90 | 23 (16) | 98 / 63% |
| Qwen3-4B | 0.692 | 0.919 | 80 [75, 85] | 97 [95, 98] | 970 / 74% | 74 | 75% | 22 / 1 / 42 / 35 | 33 (17) | 99 / 29% |
| Qwen3-8B | 0.670 | 0.875 | 67 [62, 73] | 95 [92, 97] | 999 / 79% | 62 | 58% | 22 / 3 / 55 / 21 | 52 (19) | 96 / 36% |
| Qwen3-14B | 0.659 | 0.921 | 83 [78, 87] | 98 [97, 100] | 929 / 52% | 76 | 81% | 11 / 2 / 48 / 39 | 58 (18) | 99 / 31% |
| Granite-8B | 0.682 | 0.898 | 73 [68, 78] | 78 [74, 82] | 562 / 0% | 71 | 91% | 9 / 19 / 5 / 67 | 11 (17) | 86 / 29% |

Regression (r5 formula, category for product; n = 3614, passed over 1133): best option is highest bid OR 0.29 [0.22, 0.38] (eBay 0.36 [0.28, 0.46]); finished 0.43 [0.32, 0.58]; K=7 1.95 [1.57, 2.43].
Audit nested leave-one-category-out: setting ['300|2', '300|2', '300|2'], same recoveries. At 20% calls: audit 88%, random 76%, low letter prob 81%, unfinished list 79%.
Matcher with a strictly relative 2% tolerance (no 0.05 floor): identical split.
eBay vLLM check (L40S): q(768) Phi-mini .8195 vs paper .8292, Granite-8B .8916 vs .8971; unforced 1,024 Phi-mini .8187 vs .8460, Granite-8B .8998 vs .9120.

## Overnight: closed-block readout and unforced 4,096 (an_overnight_analysis.py)

| Model | Saved letters reproduced | Open | Closed | Difference | Unforced 4,096 | Cap / median tokens |
|---|---|---|---|---|---|---|
| DeepSeek-8B | 100% | 92 [89, 95] | 95 [92, 97] | 2 [1, 4] | 99 [98, 100] | 0.9% / 1324 |
| Phi-mini | 100% | 24 [15, 32] | 81 [75, 86] | 57 [48, 66] | not run |  |
| Qwen3-4B | 100% | 80 [75, 85] | 81 [76, 87] | 2 [-2, 6] | 100 [100, 100] | 0.0% / 1474 |
| Qwen3-8B | 100% | 67 [62, 73] | 82 [78, 86] | 15 [11, 19] | 100 [100, 100] | 0.2% / 1265 |
| Qwen3-14B | 100% | 83 [78, 87] | 94 [92, 96] | 11 [8, 15] | 100 [100, 100] | 0.0% / 1047 |
| Granite-8B | 100% | 73 [68, 78] | 76 [72, 81] | 3 [1, 5] | not run |  |
