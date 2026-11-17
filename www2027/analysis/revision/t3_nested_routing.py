"""Task 3: routing with nested, grouped cross-validation.
Outer splits: (a) 6 folds grouped by value family, stratified by product; (b) leave-one-product-out.
Inner: 5-fold grouped (value family) CV on the outer-training data selects the gradient-boosting hyperparameters.
All three option orders of a scenario stay in the same fold, so no scenario appears in both training and test data.
Two routers:
  zero-cost: K, value, top-two margin, probability spread, product, budget-0 confidence, budget-0 order disagreement;
             decisions not escalated keep the budget-0 answer (no probe tokens).
  probe-128: adds 64/128-token confidence, 0-vs-64 agreement and options evaluated by 128 tokens;
             decisions not escalated keep the 128-token answer, and 128 probe tokens are charged to every decision.
Metric: area under the held-out ratio-gain-kept curve (random = 0.50) with a family-bootstrap interval holding the
outer-fold predictions fixed; plus utility ratio at the average spend of a fixed 512-token budget versus a random mix.
"""
import json, itertools, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from common_rev import *

rng = np.random.default_rng(20261113)
d = records(); g = d[d.decoding == "greedy"]
P = g[g.budget.isin([0, 64, 128, 512, 768])].pivot_table(index=["model", "scenario_id", "repeat_seed"], columns="budget",
                                                         values=["test_utility_ratio", "constrained_option_probability", "chosen_bid"]).reset_index()
P.columns = ["model", "scenario_id", "order"] + [f"{a}_{b}" for a, b in P.columns[3:]]
nd = g[g.budget == 0].groupby(["model", "scenario_id"]).chosen_bid.nunique().rename("dis0")
P = P.join(nd, on=["model", "scenario_id"])
tf = load_parse(0.02); nat = tf[tf.budget == -1].copy()
nat["ev128"] = [sum(1 for p in fp.values() if p / max(n, 1) * t <= 128) for fp, n, t in zip(nat.fp, nat.n_chars, nat.reasoning_tokens.fillna(0))]
P = P.join(nat.set_index(["model", "scenario_id", "repeat_seed"])["ev128"], on=["model", "scenario_id", "order"])
for col, fn in [("K", lambda i: S[i]["K"]), ("value", lambda i: S[i]["v"]), ("gap12", lambda i: S[i]["gap12"]),
                ("pspread", lambda i: float(S[i]["dp"].max() - S[i]["dp"].min())), ("cartier", lambda i: float(S[i]["product"].startswith("Cartier"))),
                ("xbox", lambda i: float(S[i]["product"].startswith("Xbox"))), ("fam", lambda i: S[i]["fam"]), ("product", lambda i: S[i]["product"])]:
    P[col] = P.scenario_id.map(fn)
P["agree064"] = (P.chosen_bid_0 == P.chosen_bid_64).astype(float)
ZERO = ["K", "value", "gap12", "pspread", "cartier", "xbox", "constrained_option_probability_0", "dis0"]
PROBE = ZERO + ["constrained_option_probability_64", "constrained_option_probability_128", "agree064", "ev128"]
GRID = [dict(n_estimators=n, max_depth=dpt, learning_rate=0.05, subsample=0.8) for n, dpt in itertools.product([100, 300], [1, 2, 3])]

def gain_kept_area(score, qp, q7):
    order = np.argsort(-score, kind="stable"); cum = np.r_[0, np.cumsum((q7 - qp)[order])]; n = len(score)
    grid = np.linspace(0, 1, 101); kept = np.array([cum[int(round(s * n))] / cum[-1] for s in grid])
    return float(np.trapezoid(kept, grid)), kept

def inner_select(X, yv, fams):
    uf = np.unique(fams); rng.shuffle(uf); folds = {f: i % 5 for i, f in enumerate(uf)}; fi = np.array([folds[f] for f in fams])
    best, bestp = -1e9, None
    for hp in GRID:
        sc_ = []
        for k in range(5):
            tr = fi != k
            m_ = GradientBoostingRegressor(random_state=1, **hp).fit(X[tr], yv[tr]); pr = m_.predict(X[~tr])
            sc_.append(np.corrcoef(pr, yv[~tr])[0, 1] if np.std(pr) > 0 else -1)
        if np.nanmean(sc_) > best: best, bestp = np.nanmean(sc_), hp
    return bestp

out = {}
for m in MODELS:
    x = P[P.model == m].reset_index(drop=True)
    q0, q128, q512, q7 = (x[f"test_utility_ratio_{b}"].values for b in [0, 128, 512, 768])
    fams = x.fam.values
    fold6 = {}
    for p in PRODUCTS:
        fl = list(FAM_BY_PROD[p]); rng.shuffle(fl)
        for i, f in enumerate(fl): fold6[f] = i % 6
    splits = {"family6": [(x.fam.map(fold6).values != k) for k in range(6)],
              "lopo": [(x["product"] != p).values for p in PRODUCTS]}
    res = {}
    for rname, feats, qp, cost in [("zero_cost", ZERO, q0, 0), ("probe128", PROBE, q128, 128)]:
        X = x[feats].fillna(0).values; yv = q7 - qp
        for sname, trs in splits.items():
            pred = np.full(len(x), np.nan); chosen_hp = []
            for tr in trs:
                hp = inner_select(X[tr], yv[tr], fams[tr]); chosen_hp.append(hp)
                pred[~tr] = GradientBoostingRegressor(random_state=1, **hp).fit(X[tr], yv[tr]).predict(X[~tr])
            jit = rng.uniform(0, 1e-9, len(x))
            area, kept = gain_kept_area(pred + jit, qp, q7)
            bs = []
            for _ in range(1000):
                ids = family_boot_ids(fams, rng); bs.append(gain_kept_area(pred[ids] + jit[ids], qp[ids], q7[ids])[0])
            # utility ratio at the average spend of a fixed 512-token budget
            s512 = (512 - cost) / 768; n = len(x); order = np.argsort(-(pred + jit), kind="stable")
            esc = np.zeros(n, bool); esc[order[:int(round(s512 * n))]] = True
            q_router = float(np.where(esc, q7, qp).mean())
            q_random = float(np.mean([np.where(rng.random(n) < s512, q7, qp).mean() for _ in range(200)]))
            res[f"{rname}_{sname}"] = dict(area=area, area_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                         q_at_512_spend=q_router, q_random_mix_at_512_spend=q_random, q_fixed_512=float(q512.mean()),
                                         probe_tokens=cost, hyperparams=chosen_hp)
    out[m] = res
json.dump(out, open(f"{OUTDIR}/t3_nested_routing.json", "w"), indent=1, default=float)
for m in MODELS:
    print(LAB[m], {k: (round(v["area"], 3), [round(a, 2) for a in v["area_ci"]], round(v["q_at_512_spend"], 3), round(v["q_random_mix_at_512_spend"], 3), round(v["q_fixed_512"], 3)) for k, v in out[m].items()})
