"""Task 7 support: per-decision cost, latency, energy and break-even, reproducing the cost numbers in the paper.
Rental prices: RunPod RTX 4090 $0.74/h (as in the original runtime_cost_recomputed.csv) and L40S $0.79/h (Qwen3-14B,
RunPod pricing page, accessed 2026-11-05). Break-even (B/e) = cost of 768 tokens / (q(768) - q(0)).
"""
import json, numpy as np, pandas as pd
from common_rev import *

cost = pd.read_csv("../runtime_cost_recomputed.csv").set_index("model")
d = records(); g = d[d.decoding == "greedy"]
out = {}
for m in MODELS:
    sec_tok = float(cost.loc[m, "seconds_per_reasoning_token"])
    price = 0.79 if m == "qwen3_14b" else 0.74
    lam = sec_tok * price / 3600
    c768 = 768 * lam
    q = g[g.model == m].groupby("budget").test_utility_ratio.mean()
    y = g[(g.model == m) & (g.budget == 768)]
    lat = (y.elapsed_seconds + y.constrained_readout_elapsed_seconds).mean()
    wh = (y.energy_joules + y.constrained_readout_energy_joules).mean() / 3600
    calc_u = y.scenario_id.map(lambda i: S[i]["tu"][np.isclose(S[i]["bids"], S[i]["calc"])][0])
    out[LAB[m]] = dict(seconds_per_token=sec_tok, gpu_hour_usd=price, cost_768_usd=c768, dq=float(q[768] - q[0]),
                       breakeven_usd=c768 / float(q[768] - q[0]), breakeven_at_50_per_M=768 * 50e-6 / float(q[768] - q[0]),
                       latency_s_768=float(lat), energy_wh_768=float(wh),
                       loss_vs_calculator_usd_768=float((calc_u - y.test_utility).mean()))
json.dump(out, open(f"{OUTDIR}/t7_costs.json", "w"), indent=1)
for k, v in out.items(): print(k, {a: round(b, 4) for a, b in v.items()})
