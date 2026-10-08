# Structural results for the submitted contest

6 October 2026. Algebraic properties of the specified finite game. These are not claims of a novel theorem or real-market identification.

## Normalization

With private payoff `V P_i(c_i,c_j)-lambda c_i`, dividing by the positive prize V preserves best responses and gives `P_i-kappa c_i`, with `kappa=lambda/V`. Dividing planner welfare by V gives `rho Q-kappa C`, with `rho=beta/V`. Multiplying all monetary scales by the same positive factor preserves allocations. This does not identify the monetary prize or social valuation from auction prices.

## Three allocation regions need not come from measured curve details

Consider a finite menu of at least two distinct costs/budgets, positive qualities, a strictly highest quality at the highest budget, and positive finite Tullock sensitivity. At zero cost, the highest-quality action strictly dominates every lower-quality action. By finiteness and continuity, both agents choose it for sufficiently small positive kappa. For sufficiently large kappa, the lowest-budget action strictly dominates, because contest benefits are bounded while every extra budget increment has positive cost.

For any fixed positive kappa, the planner chooses the lowest-cost pair for sufficiently small positive rho. For sufficiently large rho, it chooses the highest/highest pair: positive Tullock win probabilities imply every pair containing a lower-quality action has strictly lower expected winner quality. Combining low private cost with low social valuation gives overcompute; combining sufficiently high private cost with sufficiently high social valuation gives undercompute. High-cost/low-valuation and low-cost/high-valuation combinations give alignment. All these choices are strict on suitably chosen open parameter neighborhoods.

Thus three-region existence is generic under these sufficient conditions if the parameter domain is broad enough. The claim does not say that every bounded plotted domain contains all regions, or that every nonmonotone frontier satisfies the strict-highest condition. A flat frontier yields low-budget alignment at every strictly positive cost. Measured data contribute the numerical thresholds, uncertainty, menu-specific intermediate allocations and welfare consequences.

## Planner total compute is weakly increasing with social valuation

At fixed kappa, divide the planner objective by positive rho. It becomes `Q-r C`, with `r=kappa/rho`. Let allocations A and B maximize that objective at `r1<r2`. Optimality gives `Q_A-r1 C_A >= Q_B-r1 C_B` and `Q_B-r2 C_B >= Q_A-r2 C_A`. Adding yields `(r2-r1)(C_A-C_B)>=0`, hence `C_A>=C_B`. The proof allows asymmetric planner allocations and nonmonotone frontiers. At indifference points report the optimal set rather than hiding it behind a tie rule.

## Complete equilibria can be obtained from a zero-sum transformation

For the homogeneous symmetric Tullock game, write `P_ab+P_ba=1` and private payoffs `u1=P_ab-kappa c_a`, `u2=1-P_ab-kappa c_b`. Add the rival's cost to each player's payoff. These additions depend only on the rival's action, so they preserve best responses. The transformed payoffs are `A_ab=P_ab-kappa c_a+kappa c_b` and `1-A_ab`, a constant-sum game. Furthermore `A+A^T=1`, so `R=A-0.5` is skew symmetric and the symmetric zero-sum value is zero.

The common optimal-strategy polytope is `S={x>=0, sum(x)=1, R x<=0}`. Every Nash equilibrium is a pair `(x,y)` with `x,y in S`, and every such pair is an equilibrium. Enumerating the polytope's vertices therefore describes the full equilibrium set, including boundary multiplicity. Linear expected-compute extrema occur at vertices. Bilinear equilibrium-welfare extrema over `S x S` occur at vertex pairs. A convenient single mixed equilibrium is not the full welfare range when S is nontrivial.

This transformation applies to the submitted symmetric prize contest, not automatically to the proposed auction game, asymmetric prizes, endogenous resource cost, or distribution-sensitive outcomes. Those extensions require their own equilibrium analysis.

## CPU findings and verification

For the six submitted stochastic frontiers, sensitivity values `{0.5,1,2,4}`, and unilateral payoff-indifference cuts, the new analysis enumerated 160 open cost intervals and their boundary strategy sets after merging machine-precision duplicate intersections. One open interval has no pure equilibrium: Qwen3-4B at gamma=4, kappa approximately `(0.000328602705252,0.000328909116527)`.

At its midpoint the symmetric mixed probabilities over `{128,384,768}` are approximately `(0.00589117,0.00981862,0.98429020)`. A separate linear-programming solver agrees to within `1e-8`; expected best-response residuals in the enumerated results are below `1e-9`. Rock-paper-scissors and flat-quality dominated-action controls also pass. Exact-boundary equilibrium multiplicity is retained in `complete_equilibrium_regimes.csv`.

## Quality scale and origin

A common positive multiplicative scaling of quality cancels in Tullock success probabilities, preserving private incentives. It scales planner quality and therefore preserves planner choices only if social valuation is adjusted reciprocally. Adding a constant to quality changes relative contest advantages even though it leaves within-agent quality increments unchanged. In the CPU sensitivity, replacing q by q+0.5 reduces the maximum-budget private threshold at gamma=1 by approximately 37–39% across checkpoints. This is an origin-sensitivity diagnostic, not an economically estimated alternative map.
