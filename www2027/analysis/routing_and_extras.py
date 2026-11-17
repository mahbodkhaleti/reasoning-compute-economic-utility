"""Second batch of CPU-only analyses for the WWW 2027 short paper.

A. Compute routing between budget 0 and budget 768 (greedy, three option orders):
   - confidence rule: reason when the budget-0 option probability is below tau
   - stakes rule: reason when the best displayed expected profit S is at least tau
   - cross-fitted router: gradient boosting on prompt features plus budget-0
     confidence, trained to predict the construction-split gain of reasoning
   - random rule (reference): reason on a random share of decisions
   Thresholds are chosen on training folds using construction-split utility only
   (90% of construction gain kept, fewest tokens), and evaluated on held-out
   utility in the left-out value families (6 folds, product-stratified).
   We also report the full gain-kept vs token-share curve, scored on held-out
   utility, for each rule (no tuning) and its normalized area.
B. Family-bootstrap intervals (10,000 draws) for gap closed, return per
   compute dollar and break-even stakes; intervals for routing evaluated
   conditional on the cross-fitted rules.
C. Majority-vote economics from the five stochastic seeds.
D. Return and break-even at current list prices for API output tokens.
"""
import json, itertools, os
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

OUT = "out"; os.makedirs(OUT, exist_ok=True)
SEED = 20261106
rng = np.random.default_rng(SEED)

d = pd.read_csv("saved_records.csv.gz")
sc = pd.read_csv("scenarios.csv")
cost = pd.read_csv("runtime_cost_recomputed.csv")
lam = {m: (None if pd.isna(l) else float(l)) for m, l in zip(cost.model, cost.lambda_usd_per_token)}
label = dict(zip(cost.model, cost.model_label))
MODELS = ["deepseek_r1_llama_8b", "phi4_mini_reasoning", "qwen3_4b", "qwen3_8b", "qwen3_14b", "granite_3_2_8b_instruct"]

# ---------------- scenario features (prompt-visible only)
look, feat = {}, {}
for r in sc.itertuples():
    c = json.loads(r.candidates_json)
    look[r.scenario_id] = {float(x["bid"]): (x["train_u"], x["test_u"]) for x in c}
    tu = sorted([x["train_u"] for x in c], reverse=True)
    bids = [x["bid"] for x in c]
    feat[r.scenario_id] = dict(
        family=f"{r.category}|{r.value}", product=r.category, value=r.value, K=r.option_count,
        stakes=tu[0], rel_gap=(tu[0] - tu[1]) / tu[0] if tu[0] > 0 else 0.0,
        spread=(max(bids) - min(bids)) / r.value, max_p=max(x["train_p"] for x in c),
        calc_test=max(c, key=lambda x: x["train_u"])["test_u"])
F = pd.DataFrame.from_dict(feat, orient="index")
d["train_utility"] = [look[s][float(b)][0] for s, b in zip(d.scenario_id, d.chosen_bid)]

g = d[(d.decoding == "greedy") & d.budget.isin([0, 768])]
W = g.pivot_table(index=["model", "scenario_id", "repeat_seed"], columns="budget",
                  values=["test_utility", "train_utility", "constrained_option_probability"]).reset_index()
W.columns = ["model", "scenario_id", "order", "p0", "p768", "te0", "te768", "tr0", "tr768"]
W = W.join(F, on="scenario_id")

families = sorted(F.family.unique())
prod_of = {f: f.split("|")[0] for f in families}
# product-stratified 6-fold split of the 54 families (3 per product per fold)
fold_of = {}
for p in sorted(set(prod_of.values())):
    fl = [f for f in families if prod_of[f] == p]
    perm = rng.permutation(len(fl))
    for i, k in enumerate(perm):
        fold_of[fl[k]] = i % 6
W["fold"] = W.family.map(fold_of)
XCOLS = ["value", "K", "stakes", "rel_gap", "spread", "max_p", "p0", "is_cartier", "is_xbox"]
W["is_cartier"] = (W["product"] == "Cartier wristwatch").astype(float)
W["is_xbox"] = (W["product"] == "Xbox game console").astype(float)


def pick_threshold(score, tr0, tr768, target=0.90):
    """Escalate when score >= t. Return the t with fewest escalations that keeps
    >= target of the construction-split gain (construction utility only)."""
    gain = tr768.sum() - tr0.sum()
    order = np.argsort(-score)
    cum = np.cumsum((tr768 - tr0)[order])
    if gain <= 0:
        return np.inf
    ok = np.where(cum >= target * gain)[0]
    k = ok[0] if len(ok) else len(score) - 1
    return score[order][k]


def curve(score, te0, te768, grid=np.linspace(0, 1, 101), tie_noise=None):
    """Held-out gain kept when escalating the top share of decisions by score."""
    n = len(score)
    s = score + (tie_noise if tie_noise is not None else 0)
    order = np.argsort(-s)
    cum = np.r_[0, np.cumsum((te768 - te0)[order])]
    tot = cum[-1]
    kept = np.array([cum[int(round(q * n))] / tot for q in grid])
    return grid, kept


routing = {}
curves = []
eval_rows = []
for m in MODELS:
    x = W[W.model == m].reset_index(drop=True)
    noise = rng.uniform(0, 1e-9, len(x))
    scores_cv = {"confidence": np.full(len(x), np.nan), "stakes": np.full(len(x), np.nan),
                 "router": np.full(len(x), np.nan)}
    esc = {k: np.zeros(len(x), bool) for k in ["confidence", "stakes", "router"]}
    for k in range(6):
        tr = x.fold != k; te = ~tr
        a = x[tr]; b = x[te]
        # confidence: escalate when p0 is LOW -> score = -p0
        s_tr = -a.p0.values; t = pick_threshold(s_tr + noise[tr], a.tr0.values, a.tr768.values)
        esc["confidence"][te] = (-b.p0.values + noise[te]) >= t
        scores_cv["confidence"][te] = -b.p0.values
        s_tr = a.stakes.values; t = pick_threshold(s_tr + noise[tr], a.tr0.values, a.tr768.values)
        esc["stakes"][te] = (b.stakes.values + noise[te]) >= t
        scores_cv["stakes"][te] = b.stakes.values
        # router: predict construction gain from features, inner split for threshold
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                       subsample=0.8, random_state=SEED)
        # out-of-fold predictions on the training families for threshold choice
        inner = np.zeros(len(a))
        for j in sorted(a.fold.unique()):
            ii = (a.fold == j).values
            gb.fit(a.loc[~ii, XCOLS], (a.tr768 - a.tr0)[~ii])
            inner[ii] = gb.predict(a.loc[ii, XCOLS])
        t = pick_threshold(inner + noise[tr], a.tr0.values, a.tr768.values)
        gb.fit(a[XCOLS], a.tr768 - a.tr0)
        pb = gb.predict(b[XCOLS])
        esc["router"][te] = (pb + noise[te]) >= t
        scores_cv["router"][te] = pb
    te0, te768 = x.te0.values, x.te768.values
    full_gain = te768.mean() - te0.mean()
    res = {}
    for k in esc:
        e = esc[k]
        prof = np.where(e, te768, te0)
        res[k] = dict(escalate_share=float(e.mean()), token_saving=float(1 - e.mean()),
                      gain_kept=float((prof.mean() - te0.mean()) / full_gain))
    # curves (held-out, ranking only; scores are cross-fitted for router)
    for k in ["confidence", "stakes", "router"]:
        grid, kept = curve(scores_cv[k], te0, te768, tie_noise=noise)
        res[k]["area"] = float(np.trapezoid(kept, grid))
        res[k]["kept_at_50pct"] = float(kept[50]); res[k]["kept_at_25pct"] = float(kept[25])
        for q, v in zip(grid, kept):
            curves.append(dict(model=m, rule=k, token_share=q, gain_kept=v))
    for q in np.linspace(0, 1, 101):
        curves.append(dict(model=m, rule="random", token_share=q, gain_kept=q))
    # family bootstrap of the evaluated (fixed) rules
    fam = x.family.values
    fams_by_prod = {}
    for f in families:
        fams_by_prod.setdefault(prod_of[f], []).append(f)
    idx_by_fam = {f: np.where(fam == f)[0] for f in families}
    B = 4000
    bs = {k: [] for k in esc}
    for _ in range(B):
        ids = np.concatenate([idx_by_fam[f] for p, fl in fams_by_prod.items()
                              for f in rng.choice(fl, len(fl), replace=True)])
        g0, g7 = te0[ids].mean(), te768[ids].mean()
        for k, e in esc.items():
            bs[k].append(((np.where(e[ids], te768[ids], te0[ids]).mean() - g0) / (g7 - g0), 1 - e[ids].mean()))
    for k in esc:
        arr = np.array(bs[k])
        res[k]["gain_kept_ci"] = [float(np.percentile(arr[:, 0], 2.5)), float(np.percentile(arr[:, 0], 97.5))]
        res[k]["saving_ci"] = [float(np.percentile(arr[:, 1], 2.5)), float(np.percentile(arr[:, 1], 97.5))]
    # within-product check of stakes rule: ranking area inside each product
    within = {}
    for p in sorted(x["product"].unique()):
        ii = (x["product"] == p).values
        _, kept = curve(x.stakes.values[ii], te0[ii], te768[ii], tie_noise=noise[ii])
        _, keptc = curve(-x.p0.values[ii], te0[ii], te768[ii], tie_noise=noise[ii])
        within[p] = dict(stakes_area=float(np.trapezoid(kept, np.linspace(0, 1, 101))),
                         confidence_area=float(np.trapezoid(keptc, np.linspace(0, 1, 101))))
    res["within_product_area"] = within
    res["label"] = label[m]
    routing[m] = res
pd.DataFrame(curves).to_csv(f"{OUT}/routing_curves.csv", index=False)

# ---------------- B. bootstrap intervals for the dollar table
cell = (d[(d.decoding == "greedy") & d.budget.isin([0, 768])]
        .groupby(["model", "scenario_id", "budget"])
        .agg(U=("test_utility", "mean"), Q=("test_utility_ratio", "mean")).reset_index())
tab = {}
for m in MODELS:
    c = cell[cell.model == m]
    U = c.pivot(index="scenario_id", columns="budget", values="U")
    Q = c.pivot(index="scenario_id", columns="budget", values="Q")
    calc = F.loc[U.index, "calc_test"]
    famv = F.loc[U.index, "family"].values
    idx = {f: np.where(famv == f)[0] for f in families}
    byp = {}
    for f in families:
        byp.setdefault(prod_of[f], []).append(f)
    u0, u7, q0, q7, cu = U[0].values, U[768].values, Q[0].values, Q[768].values, calc.values
    c768 = 768 * lam[m] if lam[m] else None
    draws = []
    for _ in range(10000):
        ids = np.concatenate([idx[f] for p, fl in byp.items() for f in rng.choice(fl, len(fl), replace=True)])
        gain = u7[ids].mean() - u0[ids].mean(); dq = q7[ids].mean() - q0[ids].mean()
        gap = gain / (cu[ids].mean() - u0[ids].mean())
        draws.append((gain, dq, gap))
    a = np.array(draws)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    gain = u7.mean() - u0.mean(); dq = q7.mean() - q0.mean()
    tab[m] = dict(label=label[m], gain=float(gain), gain_ci=ci(a[:, 0]), dq=float(dq), dq_ci=ci(a[:, 1]),
                  gap=float(gain / (cu.mean() - u0.mean())), gap_ci=ci(a[:, 2]),
                  cost768=c768,
                  ret=(gain / c768) if c768 else None, ret_ci=(ci(a[:, 0] / c768) if c768 else None),
                  breakeven=(c768 / dq) if c768 else None,
                  breakeven_ci=(ci(c768 / a[:, 1]) if c768 else None))

# ---------------- C. majority vote from the stochastic seeds
s = d[d.decoding == "stochastic"]
mv_rows = []
for m in MODELS:
    for B_ in [128, 384, 768]:
        x = s[(s.model == m) & (s.budget == B_)]
        P = x.pivot(index="scenario_id", columns="repeat_seed", values="chosen_bid")
        PR = x.pivot(index="scenario_id", columns="repeat_seed", values="constrained_option_probability")
        seeds = list(P.columns)
        for k in [1, 3, 5]:
            vals_U, vals_Q = [], []
            for comb in itertools.combinations(seeds, k):
                us, qs = [], []
                for sid in P.index:
                    bids = P.loc[sid, list(comb)].values; probs = PR.loc[sid, list(comb)].values
                    cnt = {}
                    for b_, p_ in zip(bids, probs):
                        c0, s0 = cnt.get(b_, (0, 0.0)); cnt[b_] = (c0 + 1, s0 + p_)
                    win = max(cnt.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
                    tu = look[sid][float(win)][1]
                    ust = max(v[1] for v in look[sid].values())
                    us.append(tu); qs.append(tu / ust)
                vals_U.append(np.mean(us)); vals_Q.append(np.mean(qs))
            mv_rows.append(dict(model=m, label=label[m], budget=B_, k=k, tokens=k * B_,
                                q=float(np.mean(vals_Q)), profit=float(np.mean(vals_U)),
                                cost_usd=(k * B_ * lam[m]) if lam[m] else None))
MV = pd.DataFrame(mv_rows)
MV.to_csv(f"{OUT}/majority_vote.csv", index=False)

# ---------------- D. list-price sensitivity (USD per 1M output tokens, Nov 2026)
PRICES = {"$0.50/M (cheapest current tier)": 0.50, "$2.50/M": 2.50, "$12/M": 12.0, "$50/M (top tier)": 50.0}
price_rows = []
for m in MODELS:
    t = tab[m]
    for name, p in PRICES.items():
        c = 768 * p / 1e6
        price_rows.append(dict(model=m, label=label[m], price=name, cost768=c,
                               ret=t["gain"] / c, breakeven=c / t["dq"]))
PR_ = pd.DataFrame(price_rows); PR_.to_csv(f"{OUT}/list_price_sensitivity.csv", index=False)

json.dump(dict(routing=routing, table=tab), open(f"{OUT}/routing_and_extras.json", "w"), indent=1, default=float)

# ---------------- print summary
for m in MODELS:
    r = routing[m]
    print(label[m], {k: (round(r[k]["token_saving"], 3), round(r[k]["gain_kept"], 3),
                         [round(v, 2) for v in r[k]["gain_kept_ci"]], round(r[k]["area"], 3),
                         round(r[k]["kept_at_50pct"], 3)) for k in ["confidence", "stakes", "router"]})
    print("   within-product", {p[:6]: (round(v["stakes_area"], 3), round(v["confidence_area"], 3)) for p, v in r["within_product_area"].items()})
for m in MODELS:
    t = tab[m]
    print(t["label"], round(t["gain"], 2), [round(v, 2) for v in t["gain_ci"]], "gap", round(t["gap"], 3),
          [round(v, 3) for v in t["gap_ci"]], "ret", t["ret"] and round(t["ret"]), t["ret_ci"] and [round(v) for v in t["ret_ci"]],
          "be", t["breakeven"] and round(t["breakeven"], 4), t["breakeven_ci"] and [round(v, 4) for v in t["breakeven_ci"]])
print(MV.pivot_table(index="label", columns=["budget", "k"], values="q").round(3).to_string())
print(MV.pivot_table(index="label", columns=["budget", "k"], values="profit").round(1).to_string())
print(PR_.groupby("price")[["cost768", "ret", "breakeven"]].agg(["min", "max"]).round(3).to_string())
