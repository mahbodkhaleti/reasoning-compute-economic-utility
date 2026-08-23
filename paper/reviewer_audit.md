# Current submission audit (2026-08-22)

## What is now supported

The paper measures leakage-free, held-out reasoning-compute--economic-utility
frontiers for four checkpoints. Candidate construction is independent of the
optimal action, option order is randomized, closing-price scoring uses held-out
auctions, and the primary frontier includes five stochastic generations. The
hierarchical bootstrap propagates scenario and generation uncertainty through
the downstream game. Free-form readout, deterministic full-menu/frontier, and
held-out auction-CDF analyses are explicitly secondary sensitivity checks.

The strategic exercise is correctly framed as a conditional calibrated
comparative static. It does not claim that the models chose compute strategically
or that real-world overcomputation was observed. The primary strategic result is
the location of overcompute, aligned, and undercompute regions in dimensionless
$(\kappa,\rho,\gamma)$ space. Old grid shares, welfare summaries, and PoRC are
secondary declared-design diagnostics.

## High-priority sensitivity findings

- Algebra and 432 numerical comparisons confirm that Nash and planner
  allocations depend only on $\kappa=\lambda/V$, $\rho=\beta/V$, and $\gamma$.
  Normalized welfare loss agrees with legacy loss divided by $V$ to numerical
  precision. The old fixed near-zero cutoff for relative loss is the only
  non-scale-invariant reporting convention and is excluded from phase analysis.
- Every checkpoint exhibits overcompute, aligned, and undercompute regions for
  every tested $\gamma$. Private transitions are vertical in $\kappa$; planner
  transitions are diagonal because planner choice depends on $\kappa/\rho$.

- The baseline beta=1 strategic result is normalization-sensitive. Increasing
  the planner's quality scale raises planner compute and lowers excess-compute
  share; at beta=1000, excess-compute share is zero for all four checkpoints.
  Welfare loss can remain positive because the failure mode becomes equilibrium
  undercomputation.
- Varying gamma materially changes equilibrium compute and excess-compute share.
  The paper reports the full evaluated grid and does not call these statistical
  confidence intervals.
- On the deterministic six-budget frontier, excess-compute share remains between
  72.3% and 79.5% across models. This is a combined menu/frontier sensitivity,
  not a clean menu-only comparison, because decoding also changes.
- A zero-compute-augmented sensitivity combines the deterministic, order-averaged
  reasoning-mode `c=0` observation with the stored five-seed stochastic positive
  budgets. Zero is selected at sufficiently high relative compute cost and the
  exact Nash and planner thresholds move materially, but overcompute, alignment,
  and undercompute open regions survive for every model and tested gamma. This
  is explicitly labeled hybrid because the zero and positive actions are not
  protocol matched. In 1,000 bootstrap draws all three regions always remain;
  rare resampled frontiers also contain a narrow interval without a symmetric
  pure equilibrium and are flagged separately.
- Resampling held-out auctions within product leaves every model's 0-to-768
  utility gain positive. This isolates finite held-out-CDF estimation and is
  distinguished from scenario and generation bootstraps.

## Remaining limitations and unavailable claims

1. A pure six-budget stochastic comparison, or even a protocol-matched
   stochastic four-budget comparison including zero, requires new stochastic
   inference at the missing budgets; neither can be reconstructed from existing
   outputs. The hybrid sensitivity addresses the action-set concern only
   partially.
2. Alternative construction/held-out split seeds would change prompt-side CDFs
   and therefore require new candidate construction and model inference. The
   available auction bootstrap only re-scores fixed stored choices.
3. The completed runs did not contemporaneously log exact Hugging Face commits
   or exact Transformers/Accelerate versions. The paper now reports the sole
   snapshots recovered by a post-hoc cache audit and the preserved environment
   versions, explicitly conditional on cache immutability; the runner pins and
   records them prospectively.
4. The normalized phase diagram removes redundant nominal scales but does not
   identify $\kappa$, $\rho$, $\gamma$, or a market distribution over them. The
   contest mapping and economic primitives remain assumptions rather than
   estimates from strategic model interaction.
5. Direct-game pilot outputs are not used as evidence in the manuscript.
6. Product coverage is limited to the three studied categories; the design does
   not establish external validity for other auction mechanisms or deployments.
7. Code is anonymously archived under MIT and model licenses are documented.
   The source dataset mirror still reports an unknown license, so raw and
   processed auction CSVs are not redistributed and the asset-license checklist
   answer remains conservative. Total project-wide compute accounting also
   remains incomplete.

## Submission posture

The defensible contribution is the empirical measurement and uncertainty-aware
economic calibration pipeline, not a novel theory of strategic test-time
compute. Claims in the abstract, introduction, discussion, and conclusion now
follow that boundary consistently.
