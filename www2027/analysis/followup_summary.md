# GPU follow-up runs: summary

All numbers come from scripts in `analysis/followup/`:

- `prep_traces.py` runs the paper's trace extractors, `extract5.py` and `extract8.py`, unchanged.
- `run1_analysis.py` reuses the paper's text reader, hybrid written values, categories and family bootstrap, through `cpu_revisions/cr_common.py`.
- Raw outputs are in `results/followup/<run>/`.

## Hardware note (applies to every run)

The follow-up runs used one RTX 6000 Ada with the paper's library versions and pinned model revisions. The paper's runs used an RTX 4090, and an L40S for Qwen3-14B.

The drift check reran the paper's forced-768 protocol on 10 scenarios × 1 order per Run 1 model:

| Model | Traces identical to the paper, token for token | Same chosen option |
|---|---|---|
| Qwen3-8B | 10/10 | 10/10 |
| Qwen3-14B | 3/10 | 10/10 |
| DeepSeek-8B | 1/10 | 10/10 |
| Qwen3-4B | 0/10 | 8/10 |

The differences start at near-tied word choices, such as "make" versus "choose". This is rounding on a different GPU, not a protocol difference. In total, 38 of 40 choices agree. q(0), q(768) and the calculator below come from the paper's saved records.

## Run 1: natural stopping, cap 4,096

Each run has 648 decisions per model: 216 scenarios × 3 option orders, greedy decoding, the paper's prompt and readout.

Column definitions:

- **Gap recovered** is (q − q(0)) / (q_calc − q(0)), in %, with a 95% family-bootstrap interval from 2,000 draws.
- **768** and **1,024 run** are recomputed from the paper's records. They match Table 1's point estimates.
- **Finished list** is the share of traces whose written values cover every option, using the paper's hybrid written values.
- **Cost per decision** uses the paper's basis: seconds per token on the original GPUs × RunPod rate.

| Model | q(768) | q 1,024 run | q natural | Gap 768 | Gap 1,024 run | Gap natural | At cap | Tokens median / mean | Finished list | Wrong (of 648) | Cost per decision, 768 → natural | B/e 768 → natural |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | .916 | .960 | .980 | 72 [66, 78] | 91 [87, 95] | 100 [100, 100] | 1.1% | 1,390 / 1,580 | 98% | 2 | 0.22¢ → 0.45¢ | 1.3¢ → 2.0¢ |
| Qwen3-8B | .907 | .951 | .980 | 66 [60, 73] | 87 [82, 91] | 100 [100, 100] | 0.6% | 1,226 / 1,397 | 96% | 1 | 0.34¢ → 0.61¢ | 2.3¢ → 2.8¢ |
| Qwen3-14B | .916 | .954 | .980 | 72 [66, 78] | 89 [83, 93] | 100 [100, 100] | 0.0% | 980 / 1,138 | 92% | 0 | 0.75¢ → 1.11¢ | 4.5¢ → 4.8¢ |
| DeepSeek-8B | .950 | .972 | .978 | 85 [79, 91] | 96 [93, 98] | 99 [98, 100] | 0.6% | 1,327 / 1,406 | 95% | 25 | 0.31¢ → 0.57¢ | 1.8¢ → 2.8¢ |

The calculator's q is .980 for all four models.

**Table 2 split for natural stopping.** Wrong decisions are so rare that shares of the gap are not stable, so counts are given instead:

| Model | Not reached | Arith. | Incompl. | Passed |
|---|---|---|---|---|
| Qwen3-4B | 0 | 0 | 0 | 2 |
| Qwen3-8B | 1 | 0 | 0 | 0 |
| Qwen3-14B | 0 | 0 | 0 | 0 |
| DeepSeek-8B | 4 | 0 | 0 | 21 |

Measured on the RTX 6000 Ada, a natural-stopping decision took 44–63 s and 3.3–5.1 Wh. Two models shared the GPU for part of the run, so wall times are upper bounds.

The 8,192 rerun was not needed: no model had more than 10% of decisions at the cap, the highest share being 1.1%.

Figure: `results/followup/fig_main_run1.pdf` is Fig. 1a with large hollow markers for the 4,096 run, joined to the 1,024-run markers.

## Status of the paper's claims after Run 1

1. **"At 768 forced tokens the answers leave 15–34% of the gap to the calculator" (Qwen3, DeepSeek): holds.** The numbers reproduce exactly from the saved records.
2. **"The loss at 768 for Qwen3 comes mostly from traces cut off before they finish" (Not reached plus Incompl.): strengthened.** With room to finish, the Qwen3 models pick the calculator's option in 99.7–100% of decisions.
3. **The practical message weakens for these four models.** Removing the cap closes 99–100% of the gap. The text reader at 768 closes 87–96%. Natural stopping costs 1.5–2.1 times the tokens of 768: mean 1,138–1,580 tokens, 0.45–1.11¢ per decision. Its break-even is 2.0–4.8¢, against 1.3–4.5¢ at 768. For these models, "let the model finish" beats "read the 768-token trace", with no trace-reading machinery. The audit and reader argument still applies where tokens are capped by cost or latency.
4. **"Passed over": holds for DeepSeek-8B only, and the effect is small.** 21 of its 25 wrong natural-stopping decisions write every option with the best ranked first, yet the answer differs. That is 3% of decisions and about 1 point of gap.
5. **Untested by Run 1.** Phi-mini and Granite-8B, where Passed dominates at 768 (82% and 61% of the gap), were not part of Run 1. Run 4's unforced setting is the first new evidence on them, and it is capped at 1,024.

## Run 4: prompt instruction ("End by listing the expected profit of each option, then answer with the option that has the largest.")

Gap recovered (%) with 95% family-bootstrap interval. "vs paper" is the paired difference from the paper's own decisions in the same setting.

| Model | Setting | Gap with instruction | Paper | Difference | Stated answer scored | Stated agrees with letter | Wrong (paper) |
|---|---|---|---|---|---|---|---|
| phi4_mini_reasoning | forced768 | 38 [29, 45] | 42 [34, 50] | -5 [-13, 4] | 62 [55, 69] | 60% | 373 (375) |
| phi4_mini_reasoning | unforced1024 | 54 [43, 62] | 49 [41, 56] | 5 [-5, 13] | 91 [88, 95] | 58% | 294 (315) |
| granite_3_2_8b_instruct | forced768 | 68 [60, 76] | 65 [57, 73] | 3 [-4, 10] | 74 [66, 80] | 92% | 268 (301) |
| granite_3_2_8b_instruct | unforced1024 | 77 [70, 83] | 72 [65, 78] | 5 [-3, 13] | 77 [70, 83] | 100% | 251 (280) |
| deepseek_r1_llama_8b | forced768 | 82 [75, 89] | 85 [78, 91] | -3 [-8, 2] | 80 [73, 86] | 96% | 113 (89) |
| deepseek_r1_llama_8b | unforced1024 | 94 [90, 97] | 96 [93, 98] | -2 [-6, 2] | 93 [90, 96] | 97% | 57 (42) |
| qwen3_8b | forced768 | 44 [36, 52] | 66 [60, 72] | -22 [-31, -14] | 44 [36, 52] | 98% | 282 (209) |
| qwen3_8b | unforced1024 | 73 [66, 80] | 87 [82, 91] | -14 [-21, -8] | 73 [66, 80] | 100% | 137 (82) |

What this means for the paper's claims:

- **The instruction does not fix the loss.** For Phi-mini, Granite-8B and DeepSeek-8B, gap recovered changes by at most 5 points, and every interval includes zero.
- **The instruction hurts Qwen3-8B.** Its gap recovered falls by 22 points at 768 tokens and by 14 points without forcing.
- **The Phi-mini readout claim is strengthened.**
  - Even when told to answer with the largest, Phi-mini's letter agrees with the answer its trace states only 58–60% of the time.
  - Its stated answer is the best option in 96% of decisions.
  - Without forcing, scoring the stated answer recovers 91% of the gap.
- **The Granite-8B claims hold.**
  - Its stated answer agrees with its letter (92–100%).
  - Passed over is still 46–56% of its gap and arithmetic 23–32%.
