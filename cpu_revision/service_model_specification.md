# Explicit service interpretation and alternatives

8 October 2026. These are model definitions and accounting identities, not new observations of client behavior.

## Service setting

A buyer requests a bidding recommendation from two decision-service providers. Both providers choose an inference budget and incur its resource cost. The buyer selects one recommendation and pays its provider a fixed service fee V. A provider's payoff is V times its selection probability minus its own cost. The buyer's modeled benefit depends on the selected recommendation. Selection is probabilistic because comparison is noisy; the alternative rules below state precisely which information it rewards.

This setting assumes both providers incur costs before selection. A deployment where only the selected provider generates a recommendation would require a different cost model. The recommendation is evaluated against the fixed exogenous opponent/price environment in the benchmark, not an endogenous auction game.

## Accounting

For selection probabilities P1 and P2, expected client benefit B, and token costs lambda*c1 and lambda*c2:

- Client payoff: B - V.
- Provider payoffs: V*P1 - lambda*c1 and V*P2 - lambda*c2.
- Their sum: B - lambda*(c1+c2), because P1+P2=1.

The service fee is an internal transfer within this modeled client/provider group. This accounting does not include the auction seller, competing buyers, latency, environmental externalities or heterogeneous client objectives. It is therefore modeled service welfare, not total auction surplus. The historical auction closing price does not identify V.

## Quality and benefit specifications

1. **Primary utility ratio:** mean of scenario-level realized expected-profit/oracle-profit ratios. The primary contest assumes relative-quality Tullock selection and benefit beta*q. Zero and one have benchmark interpretations, but the cardinal client-value mapping and beta remain unestimated.
2. **Expected client profit:** q-dollar = mean expected bidder profit / mean oracle bidder profit. The average uses the same constructed scenario population. Define B(c) as mean expected bidder profit, so B(c) = mean oracle profit * q-dollar(c). This supplies dollar-valued client benefit under the explicit hypothesis that benchmark performance predicts service benefit. It does not supply evidence for that hypothesis. In normalized planner units, rho = mean oracle profit / V. A comparison holding the primary rho fixed would change the underlying client-value assumption and is not reported as welfare robustness.
3. **Quality-difference selection:** a client's comparison signal is the quality difference plus zero-centered logistic noise of scale s. Selection probability is logistic((q1-q2)/s). The fixed per-model reference scale is the mean of the three point utility scores. This matches the primary gamma=1 Tullock local slope at equal reference quality. Origin shifts leave selection unchanged when the scale stays fixed. Half/double discrimination checks use twice/half this noise scale. None is estimated from actual choice observations.

Accuracy-based selection is a comparator answering what happens when exact optimality is rewarded; it is not proposed as the correct client objective. Endpoint-matched linearization is a shape diagnostic with the measured utility endpoints fixed; it is not an alternative empirical frontier.

## Interpretation gates

CPU analysis can verify accounting, score units, algebraic invariance, conditional thresholds and sampling uncertainty. It cannot establish actual service fees, client-choice technology, willingness to pay, richer economic reasoning or agreement above the tested compute ceiling. Those require new external evidence or inference. The current paper should retain conditional language and distinguish these gates from protocol and numerical validation.
