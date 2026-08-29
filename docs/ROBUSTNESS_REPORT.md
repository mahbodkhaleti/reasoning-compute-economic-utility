# Robustness analysis status

Status date: 2026-08-29. This document records the analyses included in the
submission. Machine-readable provenance is in `RESULTS_PROVENANCE.md`; the
canonical command is `bash reproduce_all_from_preserved.sh`.

## Completed and reported

- Six checkpoints have complete historical deterministic, no-thinking, and
  five-seed stochastic records over the same 216 scenarios.
- Every checkpoint has 4,536 deterministic synthetic-IPV records on the shared
  exact-payoff benchmark. The benchmark hash and per-model coverage are checked
  before analysis.
- The primary 0-to-768 comparison uses 2,000 paired scenario-bootstrap draws.
  All six historical and synthetic endpoint differences are positive with the
  reported 95% intervals excluding zero.
- The stochastic frontier uses budgets 128, 384, and 768, five generation
  seeds, temperature 0.6, top-p 0.95, and a fixed option-order seed. A
  hierarchical scenario/generation bootstrap propagates uncertainty through
  the game.
- Free-form validity, conditional agreement, and available-case utility are
  reported. Granite has no paired deterministic free-form endpoint because no
  budget-zero response passed the fixed parser; nothing is imputed.
- The held-out auction-CDF bootstrap re-scores fixed choices over 2,000
  auction-level resamples without regenerating prompts or model outputs.
- Exact normalized game enumeration reports Nash and planner transitions for
  all six frontiers and every tested gamma. Overcompute, aligned, and
  undercompute open regions occur in the specified contest.
- The zero-compute sensitivity combines deterministic order-averaged zero with
  positive-budget stochastic observations. It is explicitly marked as a mixed
  protocol and has 1,000 hierarchical-bootstrap draws.
- Runtime, energy, and monetary-cost analyses use stored measurements. The
  Qwen3-14B L40S results are separated from the RTX 4090 comparison.
- Full synthetic sweep wall times and accounting costs are reported separately
  from the marginal per-reasoning-token cost used in the contest calibration.

## Validation

The release validator checks six-model identities and revisions, required
paths, Python syntax, preserved-output hashes, historical and synthetic row
coverage, duplicate keys, finite scores, prohibited raw-source files, and the
top-level SHA-256 manifest. The canonical replay additionally verifies exact
game enumeration and regenerates all reported downstream outputs without model
inference.

## Remaining limitations

1. A protocol-matched stochastic four-budget design including zero would
   require new inference; the reported zero-augmented analysis is a hybrid
   sensitivity.
2. Alternative construction/held-out split seeds would alter prompt-side CDFs
   and require new model inference.
3. Original-four revision attribution relies on the disclosed sole-cache-
   snapshot assumption. Qwen3-14B and Granite revisions were logged during the
   runs.
4. Granite's `thinking=True` template is not semantically identical to Qwen's
   native thinking mode and is treated as a protocol sensitivity.
5. The normalized contest does not identify deployed values of kappa, rho, or
   gamma, and the evaluated LLMs did not choose compute strategically.
6. The source dataset mirror reports an unknown license. The raw source is not
   redistributed; transformed audit artifacts retain that unresolved status.
7. Historical main-run and preliminary/failed-run total compute cannot be
   reconstructed, although complete synthetic-sweep time and cost are reported.

Direct prompted-game pilots remain archived but are not used as evidence in the
paper.
