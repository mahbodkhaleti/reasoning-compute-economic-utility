# What saved data exists for this paper (no new GPU inference is possible before the deadline)

All of the following are available on CPU:
- Every saved decision: model, scenario, budget (0,64,128,256,512,768 forced; unconstrained run capped at 1,024), decoding (greedy x 3 option orders; stochastic x 5 seeds at 128/384/768), chosen bid, the constrained option-letter probabilities, held-out and construction utilities, runtime, energy.
- Full reasoning text of every greedy trace (forced and no-think), option order shown to the model.
- The 216 scenarios with all candidate bids, displayed construction probabilities and held-out probabilities.
- The 628 auctions with closing price, product, auction length, number of bids/bidders and the 70/30 split (no individual bid histories).
- A synthetic independent-private-values version with exact payoffs (saved decisions for the same six models).
- Measured runtimes on RTX 4090 and L40S.
Anything requiring new model generations (new prompts, larger models, longer budgets, tool use) is NOT possible.
