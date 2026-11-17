"""Round-8 judge fixes (CPU, saved data only). Builds on t10b's hybrid written values.
C1  Null baselines for overbidding: share of non-optimal options above the calculator's bid in the same menus;
    which option wrong answers take (highest bid = highest win probability, runner-up, first/last listed, budget-0
    answer) against chance; payment and win probability for wrong answers only.
C2  Loss attribution for answers that miss a written best (Incompl. + Passed): partition of that loss, in priority
    order, into near-tie runner-up (within 2%), highest-bid option, first- or last-listed option, other; with chance
    rates. Conditional logit over the non-best options with position, win-probability rank, relative profit loss and
    budget-0 choice.
C3  Truncation: Table 2 split for K=5 only (all medians <= 656 tokens) and the passed share among complete traces.
C4  Audit: recall, share of lost value flagged, and a policy that sends flagged decisions to the calculator.
C5  Alternative win curves for scoring only (choices fixed): lognormal fit to held-out prices, and held-out prices
    shaded by 0.85 (rivals bidding below value, as in first-price markets). Gap recovered and overbid share vs the
    re-scored optimum.
C6  Interval for the complete-vs-unfinished fixed-effects contrast (family-clustered).
C7  Misc: oracle == calculator share; Granite unforced incomplete share; held-out counts by product.
"""
import json, numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
_src = open("t10b_hybrid_split.py").read()
exec(_src[:_src.index('R = {"forced768"')])
rng = np.random.default_rng(20261120)
R = {k: {} for k in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]}
held = {}
for r in sc.reset_index().itertuples():
    held[r.scenario_id] = {float(x["bid"]): x for x in json.loads(r.candidates_json)}
_tm = pd.read_csv("../trace_matches.csv", usecols=["model", "scenario_id", "repeat_seed", "budget", "option_order"])
tf = tf.merge(_tm, on=["model", "scenario_id", "repeat_seed", "budget"], how="left")
tf["order_list"] = tf.option_order.map(lambda x: [float(b) for b in json.loads(x)])
b0 = tf[tf.budget == 0].set_index(["model", "scenario_id", "repeat_seed"]).chosen_bid
z768 = tf[tf.budget == 768].copy()
z768["b0"] = [b0.get((m, i, s_), np.nan) for m, i, s_ in zip(z768.model, z768.scenario_id, z768.repeat_seed)]
def runner(i):
    s = S[i]; o = np.argsort(-s["eu"]); return float(s["bids"][o[1]])
# ---------------- C1
for m in MODELS:
    w = z768[(z768.model == m) & ~np.isclose(z768.chosen_bid, z768.calc)]
    base_over = np.mean([np.mean([b > S[i]["calc"] for b in S[i]["bids"] if not np.isclose(b, S[i]["calc"])]) for i in w.scenario_id])
    obs_over = float((w.chosen_bid > w.calc).mean())
    top = [float(max(S[i]["bids"])) for i in w.scenario_id]
    first = [o[0] for o in w.order_list]; last = [o[-1] for o in w.order_list]
    chance = lambda cond: float(np.mean([np.mean([cond(i, o, b) for b in S[i]["bids"] if not np.isclose(b, S[i]["calc"])]) for i, o in zip(w.scenario_id, w.order_list)]))
    picks = dict(
        highest_bid=(float(np.mean(np.isclose(w.chosen_bid, top))), chance(lambda i, o, b: np.isclose(b, max(S[i]["bids"])))),
        runner_up=(float(np.mean([np.isclose(c, runner(i)) for c, i in zip(w.chosen_bid, w.scenario_id)])), chance(lambda i, o, b: np.isclose(b, runner(i)))),
        first_listed=(float(np.mean(np.isclose(w.chosen_bid, first))), chance(lambda i, o, b: np.isclose(b, o[0]))),
        last_listed=(float(np.mean(np.isclose(w.chosen_bid, last))), chance(lambda i, o, b: np.isclose(b, o[-1]))),
        budget0_answer=(float(np.mean(np.isclose(w.chosen_bid, w.b0))), chance(lambda i, o, b: False)))
    pay_w = np.mean([held[i][b]["test_p"] * b for i, b in zip(w.scenario_id, w.chosen_bid)])
    pay_c = np.mean([held[i][S[i]["calc"]]["test_p"] * S[i]["calc"] for i in w.scenario_id])
    win_w = np.mean([held[i][b]["test_p"] for i, b in zip(w.scenario_id, w.chosen_bid)])
    win_c = np.mean([held[i][S[i]["calc"]]["test_p"] for i in w.scenario_id])
    lost = np.mean([held[i][S[i]["calc"]]["test_u"] - held[i][b]["test_u"] for i, b in zip(w.scenario_id, w.chosen_bid)])
    # chance of budget-0 answer among non-calc options: 1/(K-1) if b0 != calc else 0
    b0c = float(np.mean([0 if np.isclose(a, S[i]["calc"]) else 1 / (S[i]["K"] - 1) for a, i in zip(w.b0, w.scenario_id)]))
    picks["budget0_answer"] = (picks["budget0_answer"][0], b0c)
    R["C1"][m] = dict(n_wrong=int(len(w)), overbid=obs_over, overbid_chance=float(base_over), picks=picks,
                      wrong_pay=float(pay_w), calc_pay=float(pay_c), wrong_win=float(win_w), calc_win=float(win_c), wrong_lost_profit=float(lost),
                      wrong_bid_minus_calc=float(np.mean(w.chosen_bid - w.calc)))
# ---------------- C2
for m in MODELS:
    w = z768[(z768.model == m) & z768.cls.isin(["incomplete", "passed"])].copy()
    allgap = z768[z768.model == m].gap.sum()
    lab = []
    for r in w.itertuples():
        s = S[r.scenario_id]; ru = runner(r.scenario_id)
        if np.isclose(r.chosen_bid, ru) and s["gap12"] < 0.02: lab.append("near_tie_runner_up")
        elif np.isclose(r.chosen_bid, max(s["bids"])): lab.append("highest_bid")
        elif np.isclose(r.chosen_bid, r.order_list[0]) or np.isclose(r.chosen_bid, r.order_list[-1]): lab.append("first_or_last")
        else: lab.append("other")
    w["lab"] = lab
    T = w.gap.sum()
    R["C2"][m] = dict(share_of_missed={k: float(w.gap[w.lab == k].sum() / T) for k in ["near_tie_runner_up", "highest_bid", "first_or_last", "other"]},
                      count={k: int((w.lab == k).sum()) for k in ["near_tie_runner_up", "highest_bid", "first_or_last", "other"]},
                      missed_share_of_gap=float(T / allgap))
    # choice model over non-best options
    rows = []
    for g_, r in enumerate(w.itertuples()):
        s = S[r.scenario_id]; ps = dict(zip(s["bids"], s["dp"])); eu = dict(zip(s["bids"], s["eu"])); um = max(s["eu"])
        prank = {b: k / (len(s["bids"]) - 1) for k, b in enumerate(sorted(s["bids"]))}
        for j, b in enumerate(r.order_list):
            if np.isclose(b, s["calc"]): continue
            rows.append(dict(g=g_, y=int(np.isclose(b, r.chosen_bid)), pos=j / (r.K - 1), prank=prank[b],
                             loss=(um - eu[b]) / um, b0=int(np.isclose(b, r.b0)), first=int(j == 0), last=int(j == r.K - 1)))
    cd = pd.DataFrame(rows)
    cd = cd[cd.groupby("g").y.transform("sum") == 1]
    try:
        fit = ConditionalLogit(cd.y.values, cd[["loss", "prank", "first", "last", "b0"]].values, groups=cd.g.values).fit(disp=0)
        R["C2"][m]["choice_logit"] = {n: dict(coef=float(c), p=float(p)) for n, c, p in zip(["loss", "prank", "first", "last", "b0"], fit.params, fit.pvalues)}
    except Exception as e:
        R["C2"][m]["choice_logit"] = str(e)
# ---------------- C3
for m in MODELS:
    for K in (5, 7):
        z = z768[(z768.model == m) & (z768.K == K)]
        T = z.gap.sum()
        R["C3"].setdefault(m, {})[f"K{K}"] = {k: float(z.gap[z.cls == k].sum() / T) for k in cats}
    z = z768[(z768.model == m) & z768.complete]
    R["C3"][m]["complete_only_passed_share"] = float(z.gap[z.cls == "passed"].sum() / z.gap[z.cls != "right"].sum()) if (z.cls != "right").any() else None
    R["C3"][m]["complete_only_wrong_rate"] = float((z.cls != "right").mean())
# ---------------- C4
for m in MODELS:
    z = z768[z768.model == m]
    rb = np.array([pick(vv, 2, b) for vv, b in zip(z.vals, z.chosen_bid)], float)
    flag = ~np.isclose(rb, z.chosen_bid.values); wrong = ~np.isclose(z.chosen_bid.values, z.calc.values)
    qpol = np.where(flag, z.qcalc.values, z.q_read.values)
    R["C4"][m] = dict(flag_rate=float(flag.mean()), precision=float(wrong[flag].mean()), recall=float(flag[wrong].mean()),
                      false_alarm_rate=float(flag[~wrong].mean()),
                      gap_share_flagged=float(z.gap.values[flag & wrong].sum() / z.gap.values[wrong].sum()),
                      q_answer=float(z.q_read.mean()), q_policy=float(qpol.mean()), q_calc=float(z.qcalc.mean()),
                      gap_recovered_policy=float((qpol.mean() - q0[m]) / (z.qcalc.mean() - q0[m])))
# ---------------- C5
au = pd.read_csv("../auction_splits.csv")
prodmap = {}
for i in S:
    prodmap.setdefault(S[i]["product"], set())
def prod_of(item):
    it = item.lower()
    return next(p for p in PRODUCTS if p.split()[0].lower() in it)
au["product"] = au["item"].map(prod_of)
test = {p: au[(au["product"] == p) & (au.split != "train")].final_price.values for p in PRODUCTS}
R["C7"]["heldout_counts"] = {p: int(len(v)) for p, v in test.items()}
curves = {}
for p, x in test.items():
    mu, sd = np.log(x).mean(), np.log(x).std(ddof=1)
    curves.setdefault("lognormal", {})[p] = (lambda b, mu=mu, sd=sd: stats.norm.cdf((np.log(b) - mu) / sd))
    curves.setdefault("shaded085", {})[p] = (lambda b, x=x: np.mean(0.85 * x <= b))
for cname, cf in curves.items():
    for m in MODELS:
        out = {}
        for bud in (0, 768):
            z = tf[(tf.model == m) & (tf.budget == bud)]
            qs, qc, over, nw = [], [], [], 0
            for i, b in zip(z.scenario_id, z.chosen_bid):
                s = S[i]; f = cf[s["product"]]; U = {bb: f(bb) * (s["v"] - bb) for bb in s["bids"]}; um = max(U.values())
                opt = max(U, key=U.get)
                qs.append(U[float(b)] / um if um > 0 else 1); qc.append(U[s["calc"]] / um if um > 0 else 1)
                if not np.isclose(b, opt): nw += 1; over.append(b > opt)
            out[str(bud)] = dict(q=float(np.mean(qs)), qcalc=float(np.mean(qc)), overbid_of_wrong=float(np.mean(over)) if over else None)
        out["gap_recovered"] = (out["768"]["q"] - out["0"]["q"]) / (out["768"]["qcalc"] - out["0"]["q"])
        R["C5"].setdefault(cname, {})[m] = out
# ---------------- C6
y = tf[tf.budget.isin([512, 768, -1])].copy()
y["cell"] = y.model + "|" + y.scenario_id.astype(str)
y["qd"] = y.q_read - y.groupby("cell").q_read.transform("mean")
y["cd"] = y.complete.astype(float) - y.groupby("cell").complete.transform("mean")
for b_ in (512, 768, -1):
    y[f"b{b_}"] = (y.budget == b_).astype(float); y[f"b{b_}"] -= y.groupby("cell")[f"b{b_}"].transform("mean")
y = y.rename(columns={"b-1": "bm1"})
fe = smf.ols("qd ~ cd + b512 + b768 - 1", data=y).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(y.fam.astype(str) + y.model)[0]})
R["C6"] = dict(coef=float(fe.params["cd"]), ci=[float(x) for x in fe.conf_int().loc["cd"]], p=float(fe.pvalues["cd"]))
# ---------------- C7
R["C7"]["oracle_equals_calc"] = {m: float(np.isclose(z768[z768.model == m].best_eval_bid.fillna(-1), z768[z768.model == m].calc).mean()) for m in MODELS}
zn = tf[(tf.model == "granite_3_2_8b_instruct") & (tf.budget == -1)]
R["C7"]["granite_unforced_incomplete"] = float(zn.gap[zn.cls == "incomplete"].sum() / zn.gap.sum())
json.dump(R, open(f"{OUTDIR}/t11_judge8.json", "w"), indent=1, default=float)
r3 = lambda o: round(float(o), 3)
for k in R:
    print("=====", k)
    if isinstance(R[k], dict):
        for m, v in R[k].items(): print(m, json.dumps(v, default=r3))
