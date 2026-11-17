# v10 numbers and their sources

Changed or new in v10 (round-6 judge fixes). Every number was printed from the listed output.

| Where in paper | Numbers | Script → output |
|---|---|---|
| Abstract, Table 1 "Text reader ≥2", text | 97 [94,99], 91 [86,95], 88 [84,92], 88 [83,91], 91 [87,95], 72 [67,77] at 768 | extract7.py (device, raw traces) → trace_reader7.csv; t9b_trace_readers.py → t9b_trace_readers.json (gap_v_seg2, ci768) |
| Table 1 "Text reader ≥K−1" | 92, 87, 76, 74, 77, 72 | t9b_trace_readers.json (gap_v_seg) |
| Table 1 "Oracle" | 100, 98, 96, 92, 98, 81 | t9b_trace_readers.json (gap_be); same as v9 "any evaluated" |
| Text: reader at 512 (66–88%) vs answer (17–56%) | | t9b_trace_readers.json by_budget 512 |
| Text: reader never lowers q by >0.025 (≥2), >0.028 (≥K−1) | min diffs | t9b_trace_readers.json diff_by_budget |
| Text: reader validation 96% found, 95% correct (750 annotated stated values) | | t9b_trace_readers.json validation |
| Text: stated-conclusion reader worse for five models | conc gaps 71,63,46,28,51,62 vs answer | t9b_trace_readers.json gap_v_conc |
| Table 2 four-way split with CIs; abstract 56–92% | | t9_judge6.py → t9_judge6.json A2 |
| Table 2 medians (cumulative incidence, all traces) | 440/544 … 464/616 | t9_judge6.json A3 (K5/K7 median) |
| Text: completion agreement 96–99.8% | | t9_judge6.json A3 agree_by768_vs_forced |
| Figure 1b | cumulative incidence curves | t9_judge6.json A3 curve; fig_main_v10.py |
| Figure 1c | answer / text reader ≥2 / oracle | t9b_trace_readers.json; fig_main_v10.py |
| Text: 46–87% at cap; Granite stops, 90% by 669; forced within 0.02 when stopped ≤768 | | t9_judge6.json A4 |
| Text: odds ratios with Wald CIs and event counts | Phi 12 [7,21] 369; Q4 3.4 [1.3,8.8] 85; Q8 35 [6,197] 52; Q14 14 [3,54] 61; Granite 0.06 [0.03,0.13] 221; DeepSeek 0.67 [0.28,1.6] | t9_judge6.json A5 |
| Text: passed-over answers repeat budget-0 answer 32–61% vs 17% chance, 21–33% for correct | | t9_judge6.json A5 b0_repeat_* |
| Text: voting 5×128 and 2×384 vs 1×768: 0.05–0.19 below; tie rule | −0.052…−0.188; −0.063…−0.176 | t9_judge6.json A7 |
| Table 2 caption: negative per-trace gaps | 16–54 wrong traces per model | t9_judge6.json A8 |
| Text: synthetic advantage cut by about two thirds | (0.088−0.030)/0.088 | unchanged inputs, see v8_numbers.md |
| Scenario construction (values, candidate sets) | | prepare_scenarios.py in the repository (value_grid, independent_candidates) |

Numbers not listed are unchanged from v9 (v9_numbers.md) or v8 (v8_numbers.md).
