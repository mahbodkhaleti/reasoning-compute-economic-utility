# CPU revisions (November 2026)

Everything in this folder uses saved model outputs only; no model was run again. Outputs go to `results/cpu_revisions/`.

## Reproduction check

The scripts that produce Tables 1–3 were run again from `analysis/revision/`, with outputs redirected to `results/cpu_revisions/repro/` through a patched copy of `common_rev.py` kept in `repro_scripts/`. `check_tables.py` rebuilds every table cell and compares it with `paper/main.tex`. All 96 cells (Tables 1–3, six models) match at the printed precision; the result is in `results/cpu_revisions/tables_check.json`.

## Scripts

| Script | Output | Purpose |
|---|---|---|
| `cr_common.py` | none | Shared setup. Runs the paper's own code path: the 2% matcher, the text reader, hybrid written values and gap categories. |
| `extract_v_mentions.py` | `v_mentions.csv` | Runs on the raw traces (stdlib only). Flags whether each trace states v and whether it writes (v − b). |
| `a_analyses.py A1` | `A1_audit_incentives.json` | Seller revenue, bidder surplus and total per 1,000 decisions for the model, audited and calculator answers, under both pricing rules. Also gives the A7 intervals. |
| `a_analyses.py A2` | `A2_v_visibility.json` | Whether v is visible in traces; whether the reader's flag can be computed without v. |
| `a_analyses.py A3`, `a3b_passed_freetext.py` | `A3_letter_vs_freetext.json`, `A3b_passed_freetext.json` | Letter answer vs free-text answer; position test on the free-text answer. |
| `a_analyses.py A4` | `A4_cut_position.json` | Traces cut mid-option vs at a boundary. |
| `a_analyses.py A5` | `A5_natural_stopping.json` | Unforced traces that end before the 1,024-token cap. |
| `a_analyses.py A6` | `A6_audit_lofo_family.json` | Audit rule chosen on 17 value families and tested on the held-out family. |
| `a8_curves_table.py` | `A8_curves.json` | Gap recovered under each scoring curve. |

## Alternative win curves (A8)

These curves change only how the saved choices are **scored**. The models always saw the eBay empirical curve p̂, and the calculator still maximises p̂(v − b).

| Curve | Definition | Script |
|---|---|---|
| Empirical (paper) | p(b) = share of the 189 held-out closing prices ≤ b | t10_judge7.py |
| Lognormal | Lognormal fitted to held-out closing prices per product; p(b) = F(b) | t11_judge8.py (C5) |
| Shaded 0.85 | p(b) = share of held-out closing prices × 0.85 ≤ b | t11_judge8.py (C5) |
| Order statistic | Values lognormal, fitted by maximum likelihood to (closing price = second-highest of n values, n = bidders); p(b) = mean over held-out auctions of G(b)^n | t13_judge10.py (E1) |
| First-price equilibrium | Rivals with values from that G bid β(x) = x − ∫₀ˣ G^{N−1} / G(x)^{N−1}, with N = n + 1; p(b) = mean over auctions of G(β⁻¹(b))^n | t14_judge11.py (F1) |

Gap recovered at 768 forced tokens (%):

| Model | Empirical | Lognormal | Shaded 0.85 | Order statistic | First-price eq. |
|---|---|---|---|---|---|
| DeepSeek-8B | 85 | 85 | 83 | 87 | 85 |
| Phi-mini | 42 | 44 | 34 | 43 | 40 |
| Qwen3-4B | 72 | 73 | 73 | 76 | 74 |
| Qwen3-8B | 66 | 66 | 64 | 69 | 67 |
| Qwen3-14B | 72 | 72 | 72 | 76 | 75 |
| Granite-8B | 65 | 67 | 76 | 73 | 74 |

The paper reports the range over the lognormal, order-statistic and first-price-equilibrium curves, which is 40–87%. With the shaded curve included, the range is 34–87%.

## LLM annotators (B5)

TODO (author): the model names, versions and prompts of the LLM annotators are not recorded in this repository. Add them here, and put the prompts in `annotation/llm_prompts/`, before the camera-ready version.

## Robustness of the Table 2 split (from analysis/revision/t16_parser_robustness.py)

Shares of the 768-token gap (%) as Not reached / Arith. / Incompl. / Passed.

| Model | 1% tolerance | 2% (paper) | 5% tolerance | Passed only if matcher and reader both find every value |
|---|---|---|---|---|
| DeepSeek-8B | 2 / 5 / 45 / 48 | 2 / 5 / 44 / 48 | 1 / 0 / 42 / 57 | 2 / 5 / 57 / 36 |
| Phi-mini | 7 / 1 / 14 / 78 | 5 / 2 / 11 / 82 | 4 / 1 / 10 / 85 | 5 / 2 / 22 / 71 |
| Qwen3-4B | 18 / 0 / 64 / 19 | 15 / 0 / 63 / 22 | 12 / 0 / 58 / 30 | 15 / 0 / 73 / 12 |
| Qwen3-8B | 36 / 4 / 40 / 20 | 34 / 4 / 39 / 23 | 24 / 2 / 43 / 31 | 34 / 4 / 51 / 12 |
| Qwen3-14B | 22 / 4 / 58 / 16 | 20 / 3 / 59 / 18 | 12 / 0 / 59 / 29 | 20 / 3 / 70 / 7 |
| Granite-8B | 11 / 22 / 6 / 60 | 11 / 22 / 6 / 61 | 7 / 21 / 7 / 65 | 11 / 22 / 16 / 51 |
