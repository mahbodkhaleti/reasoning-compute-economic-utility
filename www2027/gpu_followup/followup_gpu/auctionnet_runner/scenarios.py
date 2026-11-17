"""Scenario builder: the eBay prepare_scenarios.py functions, called unchanged, on AuctionNet competing prices.
Only the data source, the split (by delivery period) and the per-product bid step differ."""
from __future__ import annotations
import json, math, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline", "orig"))
import prepare_scenarios as PS            # noqa: E402  (eBay code, unchanged)
from common import stable_int             # noqa: E402

def choose_step(train_prices, cfg):
    lo, hi = np.quantile(train_prices, 0.02), np.quantile(train_prices, 0.98)
    target = cfg["data"]["target_grid_points"]
    return min(cfg["data"]["step_choices"], key=lambda s: abs((hi - lo) / s - target))

def build(T, cfg, price_col="m_any_cpm", log=print):
    rows, info = [], {}
    for product in sorted(T["product"].unique()):
        if product not in set(T.loc[T.split == "train", "product"]): continue
        tr = T.loc[(T["product"] == product) & (T.split == "train"), price_col].to_numpy(float)
        te = T.loc[(T["product"] == product) & (T.split == "test"), price_col].to_numpy(float)
        step = choose_step(tr, cfg)
        values = PS.value_grid(tr, cfg["data"]["values_per_category"], step)
        info[product] = dict(step=step, n_train=len(tr), n_test=len(te), values=values,
                             grid_points=int((np.quantile(tr, .98) - np.quantile(tr, .02)) / step))
        log(f"{product}: step {step}, {len(values)} values, {len(tr)} construction / {len(te)} held-out impressions")
        for value in values:
            for K in cfg["data"]["option_sizes"]:
                for set_index in range(cfg["data"]["candidate_sets_per_value"]):
                    cseed = stable_int(cfg["seed"], product, value, K, set_index, "candidates")
                    bids = PS.independent_candidates(value, tr, K, step, cseed)
                    cands = []
                    for b in bids:
                        tp, sp = PS.empirical_probability(tr, b), PS.empirical_probability(te, b)
                        cands.append(dict(bid=b, train_p=tp, train_u=max(value - b, 0.0) * tp, test_p=sp, test_u=max(value - b, 0.0) * sp))
                    rows.append(dict(scenario_id=f"{stable_int(product, value, K, set_index, modulo=10**12):012d}", category=product,
                                     value=float(value), option_count=K, candidate_set_index=set_index, candidate_seed=cseed,
                                     train_best_sorted_position=int(np.argmax([c["train_u"] for c in cands])),
                                     test_best_sorted_position=int(np.argmax([c["test_u"] for c in cands])),
                                     candidates_json=json.dumps(cands, separators=(",", ":"))))
    S = pd.DataFrame(rows).sort_values(["category", "value", "option_count", "candidate_set_index"]).reset_index(drop=True)
    return S, info

def calculator_check(S):
    qs, rq = [], []
    for cj in S.candidates_json:
        c = json.loads(cj); tu = np.array([x["test_u"] for x in c]); calc = int(np.argmax([x["train_u"] for x in c]))
        m = tu.max(); qs.append(tu[calc] / m if m > 0 else 1.0); rq.append(np.mean(tu / m) if m > 0 else 1.0)
    return dict(q_calc=float(np.mean(qs)), q_random=float(np.mean(rq)), n=len(S))
