# C2 and C3 (CPU)

## C2: are the identical Qwen3-4B and Qwen3-14B rows in Table 1 a copy error?

No. Recomputed from the saved decision files at 768 tokens:

| Quantity | Qwen3-4B | Qwen3-14B |
|---|---|---|
| q | 0.915785 | 0.915865 |
| Answer gap recovered | 71.9% | 72.2% |
| Stated gap recovered | 72.7% | 72.8% |

The two models choose a different bid in 28% of decisions (the same bid in 71.9%), so the files are not copies. The rows match only after rounding. Nothing changes in the paper.

## C3: the "39–130×" range

| Quantity | Value |
|---|---|
| Smallest best expected profit (`ustar` minimum) | $1.7476 |
| Qwen3-4B's unrounded break-even | 1.3393¢ |
| Ratio, Qwen3-4B | 130.5 |
| Ratio, Qwen3-14B | 38.9 |

The text's 39–130 range is correct at unrounded break-evens. Dividing by the rounded 1.3¢ gives about 135, but that is a rounding artefact.

The new vLLM cost per decision will be recomputed from `results/*/runlog.jsonl` once the AuctionNet runs exist.

**Basis for the old setup:**
- seconds per token measured with HF, batch size 1, on an RTX 4090 (L40S for Qwen3-14B);
- RunPod rates of $0.74/h (RTX 4090) and $0.79/h (L40S);
- 768 tokens per decision.
