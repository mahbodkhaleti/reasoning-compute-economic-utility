"""Batch 7: CPU analyses requested by blind review round 3.
R1 readout-failure taxonomy for complete traces (free-text answer vs constrained letter; objective substitution)
R2 additive gap decomposition at 768 and a 'best evaluated option' readout counterfactual across budgets
R3 projected utility by budget from the completion curves (by K)
R4 oracle router and a cross-fitted learned router on the ratio gain
R5 estimator details: held-out sizes per product, bootstrap SE of p at candidate bids
R6 seed-level variance
"""
import json, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
exec(open("batch6_traces_v2.py").read().split("R = {}")[0])
rng = np.random.default_rng(20261111)
d = pd.read_csv("saved_records.csv.gz"); g = d[d.decoding == "greedy"].copy()
key = ["model", "scenario_id", "repeat_seed", "budget"]
tf = tf.merge(g[key + ["freeform_chosen_bid", "freeform_valid", "constrained_option_probability"]], on=key, how="left")
tf["hp_bid"] = tf.scenario_id.map(lambda i: S[i]["bids"][np.argmax(S[i]["dp"])])
tf["low_bid"] = tf.scenario_id.map(lambda i: S[i]["bids"].min())
tf["high_bid"] = tf.scenario_id.map(lambda i: S[i]["bids"].max())
# best evaluated option (by true expected profit with displayed probabilities = what the trace's numbers favor, up to 2%)
def best_eval(i, fp):
    if not len(fp): return np.nan
    s = S[i]; b = np.array([float(x) for x in fp]); u = np.array([s["dp"][np.isclose(s["bids"], x)][0] * (s["v"] - x) for x in b])
    return b[np.argmax(u)]
tf["best_eval_bid"] = [best_eval(i, fp) for i, fp in zip(tf.scenario_id, tf.fp)]
R = {}

# ---------- R1 taxonomy for complete traces whose readout is not the calculator's bid
R1 = {}
for m in MODELS:
    res = {}
    for bud in [768, -1]:
        y = tf[(tf.model == m) & (tf.budget == bud) & tf.complete]
        bad = y[~np.isclose(y.chosen_bid, y.calc)]
        ff_ok = np.isclose(bad.freeform_chosen_bid, bad.calc)
        res[bud] = dict(n_complete=int(len(y)), share_wrong=float(len(bad) / max(len(y), 1)),
                        trace_favors_calc=float(np.isclose(bad.best_eval_bid, bad.calc).mean()),
                        freeform_says_calc=float(ff_ok.mean()),
                        readout_is_highest_prob=float(np.isclose(bad.chosen_bid, bad.hp_bid).mean()),
                        readout_is_lowest_bid=float(np.isclose(bad.chosen_bid, bad.low_bid).mean()),
                        readout_is_highest_bid=float(np.isclose(bad.chosen_bid, bad.high_bid).mean()),
                        freeform_match_calc_all_complete=float(np.isclose(y.freeform_chosen_bid, y.calc).mean()),
                        readout_match_calc_all_complete=float(np.isclose(y.chosen_bid, y.calc).mean()),
                        q_freeform_complete=float(np.mean([q_of(i, b) if b == b else np.nan for i, b in zip(y.scenario_id, y.freeform_chosen_bid)])),
                        q_readout_complete=float(y.q_read.mean()))
    R1[m] = res
R["R1"] = R1

# ---------- R2 additive decomposition at 768 and best-evaluated readout curve
R2 = {}
for m in MODELS:
    y = tf[(tf.model == m) & (tf.budget == 768)].reset_index(drop=True)
    gap = (y.qcalc - y.q_read).values; tot = gap.mean()
    unf = (~y.complete).values; comp = y.complete.values
    favor = np.isclose(y.best_eval_bid.values, y.calc.values)
    ffok = np.isclose(y.freeform_chosen_bid.values, y.calc.values)
    parts = dict(
        unfinished=float(gap[unf].sum() / gap.sum()),
        complete_trace_favors_other=float(gap[comp & ~favor].sum() / gap.sum()),
        complete_readout_vs_trace=float(gap[comp & favor].sum() / gap.sum()),
    )
    # within complete & favors calc & wrong readout: how much would the free-text answer fix
    sub = comp & favor & ~np.isclose(y.chosen_bid.values, y.calc.values)
    parts["of_readout_loss_freeform_right"] = float(gap[sub & ffok].sum() / max(gap[sub].sum(), 1e-12))
    parts["p_unfinished"] = float(unf.mean()); parts["loss_given_unfinished"] = float(gap[unf].mean()) if unf.any() else 0.0
    parts["loss_given_complete"] = float(gap[comp].mean()) if comp.any() else 0.0
    fams = y.scenario_id.map(lambda i: S[i]["fam"]).values; idx = {f: np.where(fams == f)[0] for f in FAM}
    bs = {k: [] for k in ["unfinished", "complete_trace_favors_other", "complete_readout_vs_trace"]}
    for _ in range(2000):
        ids = np.concatenate([idx[f] for p in PRODUCTS for f in rng.choice([f for f in FAM if f.startswith(p)], 18, replace=True)])
        gg = gap[ids]; T_ = gg.sum()
        bs["unfinished"].append(gg[unf[ids]].sum() / T_); bs["complete_trace_favors_other"].append(gg[(comp & ~favor)[ids]].sum() / T_)
        bs["complete_readout_vs_trace"].append(gg[(comp & favor)[ids]].sum() / T_)
    parts["ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
    # counterfactual readout: best evaluated option if any evaluated, else the model's readout
    curve = {}
    for bud in [256, 512, 768, -1]:
        z = tf[(tf.model == m) & (tf.budget == bud)]
        alt = [q_of(i, b) if b == b else q for i, b, q in zip(z.scenario_id, z.best_eval_bid, z.q_read)]
        q0 = tf[(tf.model == m) & (tf.budget == 256)].q_read.mean()
        curve[bud] = dict(q_readout=float(z.q_read.mean()), q_best_eval=float(np.mean(alt)))
    q0 = g[(g.model == m) & (g.budget == 0)].test_utility_ratio.mean(); qc = y.qcalc.mean()
    for bud in curve:
        curve[bud]["gap_recovered_readout"] = float((curve[bud]["q_readout"] - q0) / (qc - q0))
        curve[bud]["gap_recovered_best_eval"] = float((curve[bud]["q_best_eval"] - q0) / (qc - q0))
    parts["curve"] = curve
    R2[m] = parts
R["R2"] = R2

# ---------- R3 projection by budget from completion curves
R3 = {}
gridB = np.arange(0, 1025, 8)
for m in MODELS:
    nat = tf[(tf.model == m) & (tf.budget == -1)]
    q0 = g[(g.model == m) & (g.budget == 0)].test_utility_ratio.mean(); qc = nat.qcalc.mean()
    res = {}
    for kk, sub in [("K5", nat.K == 5), ("K7", nat.K == 7), ("all", np.ones(len(nat), bool))]:
        x = nat[sub]
        time = np.where(x.complete, x.comp_tok, x.reasoning_tokens)
        cdf = km(time, x.complete.astype(int).values, gridB)
        q_fin = x[x.complete].q_read.mean()
        q_unf = tf[(tf.model == m) & (tf.budget == 768) & ~tf.complete & tf.scenario_id.isin(x.scenario_id)].q_read.mean()
        proj = cdf * q_fin + (1 - cdf) * q_unf
        rec = (proj - q0) / (qc - q0)
        def first(th):
            idx = np.where(rec >= th)[0]; return float(gridB[idx[0]]) if len(idx) else None
        res[kk] = dict(q_finished=float(q_fin), q_unfinished768=float(q_unf), rec_at_768=float(rec[gridB == 768][0]),
                       rec_at_1024=float(rec[-1]), budget_for_50=first(0.5), budget_for_90=first(0.9))
    R3[m] = res
R["R3"] = R3

# ---------- R4 oracle router and cross-fitted learned router on the ratio gain
R4 = {}
gq = g[g.budget.isin([0, 64, 128, 768])].pivot_table(index=["model", "scenario_id", "repeat_seed"], columns="budget",
                                                    values=["test_utility_ratio", "constrained_option_probability", "chosen_bid"]).reset_index()
gq.columns = ["model", "scenario_id", "order"] + [f"{a}_{b}" for a, b in gq.columns[3:]]
nd = g[g.budget == 0].groupby(["model", "scenario_id"]).chosen_bid.nunique().rename("dis0")
gq = gq.join(nd, on=["model", "scenario_id"])
natf = tf[tf.budget == -1].set_index(["model", "scenario_id", "repeat_seed"])[["ev128", "ev256"]]
gq = gq.join(natf, on=["model", "scenario_id", "order"])
for col, fn in [("K", lambda i: S[i]["K"]), ("value", lambda i: S[i]["v"]), ("gap12", lambda i: S[i]["gap12"]),
                ("pspread", lambda i: float(S[i]["dp"].max() - S[i]["dp"].min())), ("cartier", lambda i: float(S[i]["product"].startswith("Cartier"))),
                ("xbox", lambda i: float(S[i]["product"].startswith("Xbox"))), ("fam", lambda i: S[i]["fam"]), ("product", lambda i: S[i]["product"])]:
    gq[col] = gq.scenario_id.map(fn)
gq["agree064"] = (gq["chosen_bid_0"] == gq["chosen_bid_64"]).astype(float)
FEAT = ["K", "value", "gap12", "pspread", "cartier", "xbox", "constrained_option_probability_0", "constrained_option_probability_64",
        "constrained_option_probability_128", "dis0", "agree064", "ev128"]
def area(score, q0, q7):
    order = np.argsort(-score, kind="stable"); cum = np.r_[0, np.cumsum((q7 - q0)[order])]; n = len(score)
    gr = np.linspace(0, 1, 101); kept = np.array([cum[int(round(s * n))] / cum[-1] for s in gr]); return float(np.trapezoid(kept, gr)), kept
for m in MODELS:
    x = gq[gq.model == m].reset_index(drop=True)
    q0, q7 = x["test_utility_ratio_0"].values, x["test_utility_ratio_768"].values
    a_or, k_or = area(q7 - q0 + rng.uniform(0, 1e-9, len(x)), q0, q7)
    fold = {}
    for p in PRODUCTS:
        fl = [f for f in FAM if f.startswith(p)]; perm = rng.permutation(len(fl))
        for i, k in enumerate(perm): fold[fl[k]] = i % 6
    folds = x.fam.map(fold).values
    pred = np.zeros(len(x)); pred_lopo = np.zeros(len(x))
    for k in range(6):
        tr = folds != k
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=1)
        gb.fit(x.loc[tr, FEAT].fillna(0), (q7 - q0)[tr]); pred[~tr] = gb.predict(x.loc[~tr, FEAT].fillna(0))
    for p in PRODUCTS:
        tr = (x["product"] != p).values
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=1)
        gb.fit(x.loc[tr, FEAT].fillna(0), (q7 - q0)[tr]); pred_lopo[~tr] = gb.predict(x.loc[~tr, FEAT].fillna(0))
    a_cv, _ = area(pred + rng.uniform(0, 1e-9, len(x)), q0, q7)
    a_lopo, _ = area(pred_lopo + rng.uniform(0, 1e-9, len(x)), q0, q7)
    # bootstrap interval of the cross-fitted area
    fams = x.fam.values; idxf = {f: np.where(fams == f)[0] for f in FAM}; bs = []
    for _ in range(500):
        ids = np.concatenate([idxf[f] for p in PRODUCTS for f in rng.choice([f for f in FAM if f.startswith(p)], 18, replace=True)])
        bs.append(area(pred[ids] + rng.uniform(0, 1e-9, len(ids)), q0[ids], q7[ids])[0])
    R4[m] = dict(oracle_area=a_or, oracle_kept_at_50=float(k_or[50]), learned_area_cv=a_cv, learned_area_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                 learned_area_lopo=a_lopo)
R["R4"] = R4

# ---------- R5 estimator details
auc = pd.read_csv("auction_splits.csv")
R5 = {"heldout_n": auc[auc.split == "test"].groupby("item").size().to_dict(), "construction_n": auc[auc.split == "train"].groupby("item").size().to_dict()}
ses = []
for i in S:
    s = S[i]; pr = np.sort(auc[(auc.item == s["product"]) & (auc.split == "test")].final_price.values)
    p = np.searchsorted(pr, s["bids"], side="right") / len(pr)
    ses += list(np.sqrt(p * (1 - p) / len(pr)))
R5["se_p_median"] = float(np.median(ses)); R5["se_p_p90"] = float(np.percentile(ses, 90))
R["R5"] = R5

# ---------- R6 seed-level variance
st = d[d.decoding == "stochastic"]
R6 = {}
for m in MODELS:
    z = st[st.model == m].groupby(["budget", "repeat_seed"]).test_utility_ratio.mean().unstack()
    R6[m] = {int(b): dict(mean=float(z.loc[b].mean()), sd_across_seeds=float(z.loc[b].std())) for b in z.index}
R["R6"] = R6
json.dump(R, open("out/batch7_results.json", "w"), indent=1, default=float)
print("done")
