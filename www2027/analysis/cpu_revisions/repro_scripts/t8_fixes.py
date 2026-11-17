"""Round-5 fixes (CPU, saved data only).
F1  Unconstrained reference for all traces (including those cut at 1,024), and a three-way split of the 768-token gap
    into arithmetic / unfinished / ignored that sums to 100% (plus a residual), with family-bootstrap intervals.
    Arithmetic: the trace's own stated values (strict product matcher: '(v-b) * p = r' with p equal to the displayed
    probability, or 'X is $r' summaries) rank a different option first than the true expected profits of the same
    evaluated options. Checked against annotator-recorded values on the 150 annotated traces.
F2  Within-scenario position test: conditional logit of ignoring on the list positions of the best and runner-up
    options, grouping the three option orders of each scenario and budget; family-bootstrap intervals. Plus what
    ignored answers pick instead.
F3  Tokens-versus-value frontier: model readout vs verifier readout (best evaluated option; model answer when nothing
    is evaluated) at each budget and for the unconstrained run.
F4  Dollar vs utility-ratio gap recovered by product.
"""
import json, numpy as np, pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit
from common_rev import *

rng = np.random.default_rng(20261115)
d = records(); g = d[d.decoding == "greedy"]
tf = load_parse(0.02)
tm = pd.read_csv("../trace_matches.csv", usecols=["model", "scenario_id", "repeat_seed", "budget", "option_order", "values"])
tf = tf.merge(tm, on=["model", "scenario_id", "repeat_seed", "budget"], how="left")
tf["order_list"] = tf.option_order.map(json.loads)

def stated_values(sid, matches):
    s = S[sid]; dp = dict(zip(s["bids"], s["dp"])); out = {}
    for b, p, r, pos, kind in sorted(json.loads(matches), key=lambda z: z[3]):
        b = float(b)
        if b not in dp: continue
        if kind == "prod" and (p is None or abs(p - dp[b]) > 0.02 * dp[b] + 1e-12): continue
        out[b] = r
    return out
tf["stated"] = [stated_values(i, m) for i, m in zip(tf.scenario_id, tf["values"])]
def implied_best(sid, fp, stated):
    if not len(fp): return np.nan
    s = S[sid]; best, bv = None, -1e18
    for x in fp:
        b = float(x); val = stated.get(b, s["eu"][np.isclose(s["bids"], b)][0])
        if val > bv: best, bv = b, val
    return best
tf["implied_best"] = [implied_best(i, fp, st) for i, fp, st in zip(tf.scenario_id, tf.fp, tf.stated)]
tf["arith"] = [(ib == ib) and (be == be) and not np.isclose(ib, be) for ib, be in zip(tf.implied_best, tf.best_eval_bid)]
q0 = g[g.budget == 0].groupby("model").test_utility_ratio.mean()
R = {"F1": {}, "F2": {}, "F3": {}, "F4": {}}

# ---------------- F1
for m in MODELS:
    res = {}
    y = tf[(tf.model == m) & (tf.budget == 768)].reset_index(drop=True)
    gap = (y.qcalc - y.q_read).values
    wrong = ~np.isclose(y.chosen_bid.values, y.calc.values)
    A = wrong & y.arith.values
    U = wrong & ~A & ~y.complete.values
    I = wrong & ~A & y.complete.values & ~np.isclose(y.chosen_bid.values, y.best_eval_bid.values)
    O = wrong & ~A & ~U & ~I
    def shares(ids):
        T = gap[ids].sum(); return [gap[ids][c[ids]].sum() / T for c in (A, U, I, O)]
    pt = shares(np.arange(len(y)))
    bs = np.array([shares(family_boot_ids(y.fam.values, rng)) for _ in range(2000)])
    names = ["arithmetic", "unfinished", "ignored", "residual"]
    res["split_768"] = {n: dict(share=float(pt[j]), ci=[float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]) for j, n in enumerate(names)}
    res["arith_trace_share_768"] = float(y.arith.mean())
    # unconstrained reference, all traces
    nat = tf[(tf.model == m) & (tf.budget == -1)].reset_index(drop=True)
    qc = nat.qcalc.mean()
    def gr(z, col): return (z[col].mean() - q0[m]) / (z.qcalc.mean() - q0[m])
    bsn = []
    for _ in range(2000):
        ids = family_boot_ids(nat.fam.values, rng); z = nat.iloc[ids]; bsn.append((gr(z, "q_read"), gr(z, "q_be")))
    bsn = np.array(bsn)
    res["natural_all"] = dict(q=float(nat.q_read.mean()), mean_tokens=float(nat.reasoning_tokens.mean()), at_cap=float((nat.reasoning_tokens >= 1024).mean()),
                              gap_readout=float(gr(nat, "q_read")), gap_readout_ci=[float(np.percentile(bsn[:, 0], 2.5)), float(np.percentile(bsn[:, 0], 97.5))],
                              gap_verifier=float(gr(nat, "q_be")), gap_verifier_ci=[float(np.percentile(bsn[:, 1], 2.5)), float(np.percentile(bsn[:, 1], 97.5))])
    R["F1"][m] = res
# arithmetic check on annotated traces (annotator-recorded values)
vs = json.load(open("../validation_sample.json")); L = {}
for p in (0, 1):
    for x in json.load(open(f"/tmp/www2027/judge/label/labels_{p}.json")): L[x["item"]] = x
n_ar = 0; n_st = 0
for k, r in enumerate(vs):
    s = S[int(r["scenario_id"])]; lab = {o["label"]: o for o in L[k]["options"]}
    st = [(float(b), lab["ABCDEFG"[j]]["value"]) for j, b in enumerate(r["option_order"]) if lab["ABCDEFG"[j]]["stated"] and lab["ABCDEFG"[j]]["value"] is not None]
    if len(st) >= 2:
        n_st += 1
        by_stated = max(st, key=lambda z: z[1])[0]; by_true = max(st, key=lambda z: s["eu"][np.isclose(s["bids"], z[0])][0])[0]
        n_ar += int(not np.isclose(by_stated, by_true))
R["F1"]["annotated_arith_changes_ranking"] = dict(share=n_ar / max(n_st, 1), n=n_st)

# ---------------- F2
y = tf[tf.budget.isin([768, -1]) & tf.complete].copy()
y["ignored"] = (~np.isclose(y.chosen_bid, y.best_eval_bid)).astype(int)
def pos_of(lst, b): return lst.index(b) + 1
def runner(sid):
    s = S[sid]; o = np.argsort(-s["eu"]); return float(s["bids"][o[1]])
y["best_pos_rel"] = [(pos_of(l, b) - 1) / (k - 1) for l, b, k in zip(y.order_list, y.best_eval_bid, y.K)]
y["runner_pos_rel"] = [(pos_of(l, runner(i)) - 1) / (k - 1) for l, i, k in zip(y.order_list, y.scenario_id, y.K)]
y["grp"] = y.model + "|" + y.scenario_id.astype(str) + "|" + y.budget.astype(str)
for m in MODELS:
    z = y[y.model == m].reset_index(drop=True)
    var = z.groupby("grp").ignored.transform(lambda s: s.nunique() > 1)
    zz = z[var].reset_index(drop=True)
    res = dict(n_complete=int(len(z)), n_informative=int(len(zz)), n_groups=int(zz.grp.nunique()))
    if len(zz) > 20:
        fit = ConditionalLogit(zz.ignored.values, zz[["best_pos_rel", "runner_pos_rel"]].values, groups=pd.factorize(zz.grp)[0]).fit(disp=0)
        res["or_best_first_to_last"] = float(np.exp(fit.params[0])); res["or_runner_first_to_last"] = float(np.exp(fit.params[1]))
        res["p_best"] = float(fit.pvalues[0]); res["p_runner"] = float(fit.pvalues[1])
        bs = []
        famz = zz.scenario_id.map(lambda i: S[i]["fam"]).values
        for _ in range(300):
            ids = family_boot_ids(famz, rng); w = zz.iloc[ids].copy()
            w["grp2"] = w.grp + "|" + pd.Series(np.arange(len(w))).astype(str).values  # keep resampled copies as separate groups
            w["grp2"] = w.grp + "|" + pd.Series(ids).map(lambda _: "").values
            try:
                f2 = ConditionalLogit(w.ignored.values, w[["best_pos_rel", "runner_pos_rel"]].values, groups=pd.factorize(w.grp)[0]).fit(disp=0)
                bs.append(np.exp(f2.params))
            except Exception:
                pass
        bs = np.array(bs)
        if len(bs): res["or_best_ci"] = [float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))]; res["or_runner_ci"] = [float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))]
    ign = z[z.ignored == 1]
    def share(fn): return float(np.mean([fn(r) for r in ign.itertuples()])) if len(ign) else None
    res["ignored_pick"] = dict(
        n=int(len(ign)),
        highest_prob=share(lambda r: np.isclose(r.chosen_bid, S[r.scenario_id]["hp"])),
        lowest_bid=share(lambda r: np.isclose(r.chosen_bid, S[r.scenario_id]["bids"].min())),
        highest_bid=share(lambda r: np.isclose(r.chosen_bid, S[r.scenario_id]["bids"].max())),
        first_listed=share(lambda r: np.isclose(r.chosen_bid, r.order_list[0])),
        last_listed=share(lambda r: np.isclose(r.chosen_bid, r.order_list[-1])),
        runner_up=share(lambda r: np.isclose(r.chosen_bid, runner(r.scenario_id))),
        last_evaluated=share(lambda r: np.isclose(r.chosen_bid, float(max(r.fp.items(), key=lambda kv: kv[1])[0]))),
        first_evaluated=share(lambda r: np.isclose(r.chosen_bid, float(min(r.fp.items(), key=lambda kv: kv[1])[0]))))
    R["F2"][m] = res

# ---------------- F3 frontier
for m in MODELS:
    pts = []
    for bud in [0, 64, 128, 256, 512, 768, -1]:
        z = tf[(tf.model == m) & (tf.budget == bud)]
        pts.append(dict(budget=bud, tokens=float(z.reasoning_tokens.fillna(0).mean()) if bud == -1 else float(bud),
                        q_readout=float(z.q_read.mean()), q_verifier=float(z.q_be.mean()),
                        gap_readout=float((z.q_read.mean() - q0[m]) / (z.qcalc.mean() - q0[m])),
                        gap_verifier=float((z.q_be.mean() - q0[m]) / (z.qcalc.mean() - q0[m]))))
    R["F3"][m] = pts

# ---------------- F4 dollar vs q by product (768 readout)
for m in MODELS:
    z = g[(g.model == m) & g.budget.isin([0, 768])].copy()
    z["product"] = z.scenario_id.map(lambda i: S[i]["product"].split()[0])
    z["calc_u"] = z.scenario_id.map(lambda i: S[i]["tu"][np.isclose(S[i]["bids"], S[i]["calc"])][0])
    z["qc"] = z.scenario_id.map(lambda i: q_of(i, S[i]["calc"]))
    res = {}
    for p, w in list(z.groupby("product")) + [("all", z)]:
        a, b = w[w.budget == 0], w[w.budget == 768]
        res[p] = dict(dollar=float((b.test_utility.mean() - a.test_utility.mean()) / (a.calc_u.mean() - a.test_utility.mean())),
                      ratio=float((b.test_utility_ratio.mean() - a.test_utility_ratio.mean()) / (a.qc.mean() - a.test_utility_ratio.mean())),
                      dollar_gap_at_0=float(a.calc_u.mean() - a.test_utility.mean()))
    R["F4"][m] = res
json.dump(R, open(f"{OUTDIR}/t8_fixes.json", "w"), indent=1, default=float)
for m in MODELS:
    r = R["F1"][m]; print(LAB[m], {k: (round(v["share"], 3), [round(a, 2) for a in v["ci"]]) for k, v in r["split_768"].items()}, "arith traces", round(r["arith_trace_share_768"], 3))
    print("   natural", {k: (round(v, 3) if isinstance(v, float) else [round(a, 2) for a in v]) for k, v in r["natural_all"].items()})
    f2 = R["F2"][m]; print("   F2", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f2.items() if k != "ignored_pick"})
    print("   picks", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f2["ignored_pick"].items()})
    print("   F3", [(p["budget"], round(p["q_readout"], 3), round(p["q_verifier"], 3)) for p in R["F3"][m]])
    print("   F4", {p: (round(v["dollar"], 2), round(v["ratio"], 2), round(v["dollar_gap_at_0"], 1)) for p, v in R["F4"][m].items()})
print(R["F1"]["annotated_arith_changes_ranking"])
