"""Pay-your-bid surplus loss at 768 tokens (saved letters) as a percentage of the calculator's surplus, plus the share of
that loss from answers that are the menu's highest bid, and the mean overbid of those answers relative to the calculator's bid.
Same surplus definition as r1_closed_downstream.py (HELD test_p x (v - b)). Writes results/followup/econ_surplus_pct.json."""
import sys, json, numpy as np
sys.path.insert(0, "/tmp/www2027/analysis/rev25")
from r_common import *
R = {}
z7 = tf[tf.budget == 768]
for m in MODELS:
    z = z7[z7.model == m]
    sm = np.array([HELD[i][float(b)]["test_p"] * (S[i]["v"] - float(b)) for i, b in zip(z.scenario_id, z.chosen_bid)])
    sc = np.array([HELD[i][float(S[i]["calc"])]["test_p"] * (S[i]["v"] - float(S[i]["calc"])) for i in z.scenario_id])
    top = np.array([np.isclose(b, max(S[i]["bids"])) and not np.isclose(b, S[i]["calc"]) for i, b in zip(z.scenario_id, z.chosen_bid)])
    loss = sc - sm
    over = [(float(b) - float(S[i]["calc"])) / float(S[i]["calc"]) for i, b, t in zip(z.scenario_id, z.chosen_bid, top) if t]
    R[m] = dict(loss_pct=float(loss.sum() / sc.sum()), loss_per_1000=float(1000 * loss.mean()), share_loss_from_highest=float(loss[top].sum() / loss.sum()),
                n_highest=int(top.sum()), mean_overbid_pct_vs_calc=float(np.mean(over)) if over else None)
json.dump(R, open("/tmp/www2027/results/followup/econ_surplus_pct.json", "w"), indent=1)
for m, r in R.items(): print(LAB[m], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
