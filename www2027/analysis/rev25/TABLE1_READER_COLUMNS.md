# Table 1, reader columns (moved out of the paper in v41)

Gap to the calculator recovered at 768 forced tokens, % [95% family-bootstrap interval], eBay.
Text reader: the option the no-arithmetic text reader picks, with its settings chosen on these traces (in-sample).
Oracle reader: among options whose true expected profit appears in the trace, the one with the highest true value (uses the answer key; an upper bound).

| Model | Text reader | Oracle reader |
|---|---|---|
| DeepSeek-8B | 96 [94,99] | 100 [100,100] |
| Phi-mini | 89 [85,93] | 98 [97,99] |
| Qwen3-4B | 88 [84,92] | 96 [95,98] |
| Qwen3-8B | 87 [83,90] | 92 [89,95] |
| Qwen3-14B | 89 [86,93] | 98 [96,100] |
| Granite-8B | 73 [66,78] | 81 [75,87] |
