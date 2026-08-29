# Non-GPU validation record

Completed during package construction:

- parsed all 38 new Python sources with `ast.parse`;
- validated every JSON configuration and metadata document;
- checked every shell script with `bash -n`;
- regenerated and byte-compared the canonical synthetic benchmark;
- validated 216 unique scenarios and balanced 54/54/54/54 design cells;
- checked the historical scenario/split SHA-256 manifest;
- generated 16,848 historical and 9,072 synthetic expected-cell manifest rows;
- exercised both synthetic analysis and normalized game enumeration on a complete 9,072-row mock result set;
- exercised deterministic, stochastic, free-form, CDF bootstrap, confidence/adaptive, primary strategic, and hybrid zero-compute analyses on saved-result-derived two-model fixtures;
- confirmed exact normalized-game agreement (maximum numerical error below `3e-15` in the fixture tests);
- exercised paper-ready CSV/LaTeX generation on the fixture outputs;
- exercised Qwen-family scale-table and PDF/PNG generation from saved 4B/8B rows plus a test fixture;
- tested verifier behavior and all expected counts before inference;
- unit-tested corrupt-row quarantine, last-row duplicate compaction, and resumable-key handling for both runners;
- checked CLI help for the production runners and validators.

No CUDA inference was performed during these local construction checks, and no
model outputs were fabricated. Completed Qwen3-14B, Granite, and six-model
synthetic outputs produced later on the GPU servers are preserved at the
repository-level paths documented in `../../docs/RESULTS_PROVENANCE.md`; the
package's `results/` directory remains empty so that a fresh server rerun cannot
silently mix with the archived records. A GPU smoke test is the required first
step only when rerunning inference from this package.

For the final submission release, the canonical repository-level offline replay
also completed from the preserved records, including all bootstrap, strategic,
zero-compute, runtime, and synthetic-IPV analyses. That validation performs no
new inference.
