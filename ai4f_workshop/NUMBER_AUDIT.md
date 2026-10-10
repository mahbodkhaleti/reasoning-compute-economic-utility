# Number audit

`verify_numbers.py` passed all 21 compact-evidence checks with **no discrepancies**. Previous-version preservation is checked separately in `technical_review.json`; it is not counted as a saved-number check.

- Abstract is byte-identical to v3 (161 words), including six checkpoints, 216 tasks, 0/768 budgets, 42–85% aggregate gap closure, and the $0.03–$0.18 / $0.02–$0.12 cost-scenario ranges.
- No empirical number was added to the introduction or the new method/contract text. New bibliography dates, pages, and identifier digits describe references rather than our experiments.
- All original main/appendix table assets are byte-identical. Endpoint, accuracy, confidence-interval, calculator-audit and resource rows match the original source tables and saved CSVs. New table placement does not change values or captions.
- Calculator mean held-out profit $212.66, regret $3.09, optimum 155/216, q=0.981364, rounding sensitivity and all six simultaneous-band rows agree with evidence.
- Displayed-optimal 42.1–86.3%, both-suboptimal 11.9–52.0%, displayed-optimal/held-out-suboptimal 9.4–22.8%, and calculator-beating 2.2–8.0% ranges match.
- Adjacent contrasts 9/30, zero opposite-direction contrasts significant in both directions after Holm over 60 tests, positive-gain concentration 80.0–86.7% in the largest 22 tasks, and free-form validity 29.1–96.7% match saved analyses.
- Historical 50,544 records, 8,424 per checkpoint and all reported resource-table values match the counters. Resource exclusions are retained.
- Summary break-even ranges at gamma=1, Qwen3-14B separate L40S scenario, and 30.40-fold product sensitivity match the original saved cost evidence. All original service-contest numeric text and proofs are byte-preserved.
- All 11 numbered equation blocks and original proposition numbering remain unchanged. New source-grammar fixtures and engineering test counts are not manuscript performance results.
- No GPU inference, scientific experiment, held-out policy rescore or invented abstention payoff was used.

Machine-readable evidence: `number_audit.json`, the source hashes and preserved-asset hashes under `audit_sources/`.
