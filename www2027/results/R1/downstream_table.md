# Closed-block letters: downstream analyses (768-token eBay traces)

Open = saved letters (reproduces r4/r5 exactly); Closed = R1 letters. Script: analysis/followup/r1_closed_downstream.py; outputs results/R1/downstream_{open,closed}.json.

| Model | Wrong answers (open / closed) | Highest bid, % of wrong (open / closed; chance) | Surplus loss per 1,000, pay-your-bid (open / closed) | Split NR/Ar/Cut/Pass open | Split closed | Audit nested LPO (open / closed) | Audit family 6-fold (open / closed) |
|---|---|---|---|---|---|---|---|
| DeepSeek-8B | 89 / 73 | 44 / 38 (18) | $22.6k / $14.6k | 2/5/44/48 | 4/8/46/42 | 100 / 100 | 100 / 100 |
| Phi-mini | 375 / 152 | 22 / 42 (18) | $33.7k / $23.0k | 5/2/11/82 | 8/3/27/62 | 98 / 99 | 97 / 99 |
| Qwen3-4B | 187 / 164 | 40 / 45 (18) | $30.5k / $37.6k | 15/0/63/22 | 19/0/62/19 | 98 / 98 | 98 / 98 |
| Qwen3-8B | 209 / 155 | 48 / 42 (18) | $30.4k / $30.8k | 34/4/39/23 | 37/4/52/7 | 96 / 97 | 96 / 97 |
| Qwen3-14B | 151 / 99 | 48 / 38 (19) | $36.0k / $25.9k | 20/3/59/18 | 32/2/61/5 | 99 / 100 | 99 / 100 |
| Granite-8B | 301 / 282 | 20 / 21 (17) | $15.1k / $10.1k | 11/22/6/61 | 13/25/4/58 | 84 / 84 | 84 / 84 |

Regression (n = 3,519 traces; passed over 1074 open / 695 closed): best option is the highest bid, OR 0.36 [0.28, 0.46] open, 0.33 [0.23, 0.46] closed. Finished list 0.38 [0.29, 0.51] / 0.40 [0.30, 0.53]; K=7 1.89 [1.48, 2.41] / 1.82 [1.36, 2.44].
Audit at 20% call rate (mean over six models): 89% open, 95% closed (random 74% / 84%).
