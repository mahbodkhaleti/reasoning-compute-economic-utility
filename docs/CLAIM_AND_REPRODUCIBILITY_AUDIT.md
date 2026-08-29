# Claim and reproducibility audit

## Claim-boundary revisions

The manuscript now separates three objects consistently:

1. measured choices under externally assigned reasoning budgets;
2. the resulting checkpoint-specific utility frontiers; and
3. allocations implied when those frontiers are inserted into the specified
   reduced-form contest.

The abstract now states the directly measured endpoint gains, notes that
intermediate budgets need not improve monotonically, and identifies the game
results as model implications. The introduction no longer assumes that extra
compute always improves an action, and the contribution statement ties the
allocation regions to the posited contest. The strategic-results section marks
the comparative statics as conditional on that mapping. Discussion and
Conclusion distinguish within-task robustness from external validity and state
that the evaluated LLMs never chose compute strategically. They also make clear
that the experiment does not identify rho, gamma, or a market distribution over
the parameter space. Strong numerical statements about observed endpoint gains,
validity, scores, and enumerated thresholds remain direct.

## Provenance decision

Qwen3-14B and Granite-8B record requested and resolved model and tokenizer
revisions contemporaneously. The original
four historical records omitted commit fields. A 2026-08-23 inspection found
one cached snapshot for each configured repository; later synthetic-run
manifests resolved the same commits contemporaneously:

| Model | Revision | Historical provenance |
|---|---|---|
| Phi-mini | `0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a` | sole inspected historical cache snapshot; corroborated by later pinned synthetic run |
| Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` | same |
| DeepSeek-8B | `6a6f4aa4197940add57724a7707d069478df56b1` | same |
| Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | same |
| Qwen3-14B | `40c069824f4251a91eefaf281ebe4c544efd3e18` | requested and resolved in historical run manifests |
| Granite-8B | `610d8c6ee9c84ce51f6dfd7bc5c0215d95d49695` | requested and resolved in historical run manifests |

The sole-snapshot evidence is not described as contemporaneous proof. Its
cache-immutability assumption and the unlogged historical chat-template
fallback status remain explicit in the manuscript and manifest.

## Checklist decision

The reproducibility answer is changed to **Yes**. This is justified because the
main reported results can be regenerated from the preserved outputs, pinned
scenario/split artifacts, code, explicit analysis seeds, and documented exact
or retrospectively recovered revisions. The retrospective provenance caveat
affects certainty about re-running the original-four inference state, but it
does not prevent independent replay of the reported measurements from preserved
generations. The missing historical `PYTHONHASHSEED` affected only descriptive
cell-level intervals in the old complete-frontier script; the main endpoint and
strategic analyses use explicit seeds. The canonical replay substitutes a
stable hash for that non-primary diagnostic and records the change.

Exact checklist justification:

> Appendix F and the anonymous repository provide a machine-readable manifest,
> prompts, code and data hashes, seeds, decoding and readout settings,
> environment records, preserved outputs, and one offline command that
> regenerates the reported analyses. Qwen3-14B and Granite-8B revisions were logged
> contemporaneously; the original four were recovered retrospectively as the
> sole snapshots in each inspected cache and are reported with that provenance
> and assumption.
