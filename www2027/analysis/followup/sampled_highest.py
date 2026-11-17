"""Highest-bid share of wrong answers under sampled decoding (saved stochastic records, 5 seeds, budget 768) vs greedy."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/tmp/www2027/analysis/rev25")
from r_common import S, MODELS, LAB
d = pd.read_csv("/tmp/www2027/analysis/saved_records.csv.gz")
R = {}
for dec in ("stochastic", "greedy"):
    z = d[(d.decoding == dec) & (d.budget == 768)]
    for m in MODELS:
        w = z[z.model == m]; w = w[[not np.isclose(b, S[int(i)]["calc"]) for i, b in zip(w.scenario_id, w.chosen_bid)]]
        hi = np.mean([np.isclose(b, max(S[int(i)]["bids"])) for i, b in zip(w.scenario_id, w.chosen_bid)])
        R[f"{dec}|{m}"] = dict(n=int((z.model == m).sum()), n_wrong=len(w), highest_share=float(hi), q=float(z[z.model == m].test_utility_ratio.mean()))
json.dump(R, open("/tmp/www2027/results/followup/sampled_highest.json", "w"), indent=1)
for k, v in R.items(): print(k, v["n"], v["n_wrong"], round(100 * v["highest_share"]), round(v["q"], 3))
