# CPU reference: conservative admission and decision core

This is a standard-library-only reference for Section 6 / Appendix B, not a GPU runner, unrestricted semantic parser, calibrated interval estimator or independently evaluated service.

## Public input contract

Construct inputs with `Manifest.create`. Supply a complete, nonempty, fixed population with unique public record IDs, immutable source strings, a declared currency, an exact nonnegative private value and a nonempty distinct feasible bid menu. All bids must be between zero and value. Resolve relevant-record inclusion publicly before constructing the manifest. The implementation cannot establish whether omitted real-world records exist or whether the declared schema reflects the intended economic quantity.

The supported full-record grammar is:

```
record_id=<ID>; price_to_beat=<amount> <currency>
```

IDs use ASCII letters, digits, underscore and hyphen. Currency uses three uppercase ASCII letters and must match the manifest. Amounts use finite nonnegative decimal strings with a period; no exponent, grouping comma, sign, currency conversion or implicit correction is supported. Whitespace and labels must match the frozen grammar. Unsupported text remains unknown. A recognized source ID or currency that conflicts with the manifest blocks a conditional decision, even for a singleton menu.

These restrictions are deliberate. They establish consistency with a declared structured source, not its factual truth, and do not resolve arbitrary narratives. The same pure parser is available as `source_price`; retain it as a comparator because it can read all fields this profile admits without an LLM.

## Proposal and transaction contract

A proposal has exactly `id`, `start`, `end`, `amount`, `currency`. Offsets are Python string character offsets, zero based, with an exclusive end, and identify the entire amount lexeme. `amount` is an exact decimal string. No boolean, floating-point number, NaN, infinity or model-confidence field is accepted.

`State.admit_batch` checks all declared chunk IDs exactly once. It verifies the source grammar and source offsets independently, then checks numeric equality, identity, units and nesting. If any item fails, the entire batch leaves the previous state unchanged and records a rejection. A batch need not cover the entire population; it must cover the selected chunk exactly. Every manifest record remains in the denominator. Unknown prices retain `[0, infinity)`. Point refinements never overwrite a conflicting prior interval.

`State.decision` returns either `conditional_certificate` or `unresolved`. The former certifies construction regret only conditional on a complete correct source contract and interval containment. It says nothing about held-out profit or real-world correctness. With a singleton action or an otherwise small bound, a decision can be certified without identifying every price; unresolved extraction fields are still retained and reported.

## Arithmetic, repair priority and budgets

Arithmetic uses `fractions.Fraction`. Bound calculations follow inclusive `price <= bid`. The certificate excludes self-comparison; minimum certificate and lowest-bid tie rules match the manuscript. Chunk sensitivity uses the sum of per-record maximum potential utility uncertainty, divided by the full population size. Chunks must form a disjoint, complete fixed partition; positive exact cost estimates are supplied from a future timing pilot. Ties use lexical public chunk IDs. The priority function does not predict expected improvement or choose an optimal schedule.

`BudgetLedger` is a sequential output-token/call-cap ledger. A call must reserve its configured output cap and consume a slot before execution. Calls cannot overlap. Failed calls retain their tokens and supplied stage-time counters. If token use is unknown, the entire reserved cap is charged conservatively and marked unknown; that is not a measured token count. Invalid runtime reports do not release a reservation. A future runner must finalize failures, preserve pending-call records after interruption, supply stage timing, include input/verification/host resources, and implement the frozen initial/repair prompts. Set the total call allowance to cover the initial extraction plus the separate repair limit. `run_policy` supplies sequential CPU orchestration with per-chunk and total repair allowances, priority selection and stopping. It expects a caller-provided model callback returning proposals, token counts and stage counters; no model backend or GPU timing data is supplied. Runtime metadata is validated before proposal admission. Callback failure or invalid runtime metadata retains the unknown-token charge at the reserved cap. Callback wall time and verification CPU time are separate diagnostic counters and must not be naively added to overlapping generation timers.

## Verification and boundaries

`test_reference.py` contains contract, transaction, cap and arithmetic integration fixtures. It cross-checks the core bounds against the separately written mathematical fixture implementation. `../check_method_bounds.py` checks exact-rational finite algebra and a failed-containment negative control. Test counts are engineering checks, not scientific sample sizes.

`check_saved_protocol.py` applies only the pure grammar parser to stripped public texts from the already reused development families. Hidden `active` and `bid` fields are absent from its input. All old narrative texts are unsupported. This documents protocol incompatibility; it is neither a baseline-performance estimate nor independent validation, and no source text is rewritten to make it pass.
