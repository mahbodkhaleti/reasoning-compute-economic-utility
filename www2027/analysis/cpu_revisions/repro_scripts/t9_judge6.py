"""Round-6 judge fixes (CPU, saved data only).
A1  Stated-value verifier: argmax of the expected profits the trace itself states (strict matcher: '(v-b) * p = r'
    with p equal to the displayed probability, or 'Option X is $r'), using only the menu shown to the model, never the
    true values. Variants: any stated value / at least K-1 stated. Gap recovered by budget, family-bootstrap CIs at 768.
A2  Split 'unfinished' into best option already evaluated before the cut-off vs never reached.
A3  Completion by token t as cumulative incidence over all unconstrained traces (stopping without finishing = never
    finishes; no censoring before the 1,024 cap). Compared with forced-768 completion.
A4  Unconstrained run: token distribution, share at the cap, and q by stopping group vs the forced-768 decision on the
    same scenario and option order.
A5  Conditional-logit odds ratios with Wald 95% CIs and event counts; do passed-over answers repeat the budget-0 choice?
A7  Majority vote: tie rule (count, then summed option probability); add 5x128 vs 1x768 sampled.
A8  Why a gap share can have a negative lower bound (traces whose wrong answer beats the calculator on held-out data).
"""
import json, itertools, numpy as np, pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit
from common_rev import *

rng = np.random.default_rng(20261116)
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
        out[b] = r          # last statement wins
    return out
tf["stated"] = [stated_values(i, m) for i, m in zip(tf.scenario_id, tf["values"])]
tf["n_stated"] = tf.stated.map(len)
def sv_pick(row, need):
    if row.n_stated >= need and row.n_stated > 0:
        return max(row.stated.items(), key=lambda kv: kv[1])[0]
    return row.chosen_bid
for name, need in [("sv_any", 1), ("sv_km1", None)]:
    tf[name] = [sv_pick(r, (r.K - 1) if need is None else need) for r in tf.itertuples()]
    tf["q_" + name] = [q_of(i, b) for i, b in zip(tf.scenario_id, tf[name])]
q0 = g[g.budget == 0].groupby("model").test_utility_ratio.mean()
R = {"A1": {}, "A2": {}, "A3": {}, "A4": {}, "A5": {}, "A7": {}, "A8": {}}

def gap_rec(z, col, m): return (z[col].mean() - q0[m]) / (z.qcalc.mean() - q0[m])

# ---------------- A1
for m in MODELS:
    rows = []
    for bud in [0, 64, 128, 256, 512, 768, -1]:
        z = tf[(tf.model == m) & (tf.budget == bud)]
        rows.append(dict(budget=bud, stated_cov=float((z.n_stated / z.K).mean()), parser_cov=float((z.n_eval / z.K).mean()),
                         q_read=float(z.q_read.mean()), q_sv_any=float(z.q_sv_any.mean()), q_sv_km1=float(z.q_sv_km1.mean()),
                         q_parser=float(z.q_be.mean()),
                         gap_read=float(gap_rec(z, "q_read", m)), gap_sv_any=float(gap_rec(z, "q_sv_any", m)),
                         gap_sv_km1=float(gap_rec(z, "q_sv_km1", m)), gap_parser=float(gap_rec(z, "q_be", m))))
    z = tf[(tf.model == m) & (tf.budget == 768)].reset_index(drop=True)
    bs = []
    for _ in range(2000):
        w = z.iloc[family_boot_ids(z.fam.values, rng)]
        bs.append([gap_rec(w, c, m) for c in ("q_sv_any", "q_sv_km1", "q_be")])
    bs = np.array(bs)
    R["A1"][m] = dict(by_budget=rows, ci768={c: [float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]
                                              for j, c in enumerate(["sv_any", "sv_km1", "parser"])},
                      sv_any_minus_read_min=float(min(r["q_sv_any"] - r["q_read"] for r in rows)),
                      sv_km1_minus_read_min=float(min(r["q_sv_km1"] - r["q_read"] for r in rows)))

# ---------------- A2
calcpos = lambda sid, fp: any(np.isclose(float(x), S[sid]["calc"]) for x in fp)
tf["calc_reached"] = [calcpos(i, fp) for i, fp in zip(tf.scenario_id, tf.fp)]
def implied_best(sid, fp, stated):
    if not len(fp): return np.nan
    s = S[sid]; best, bv = None, -1e18
    for x in fp:
        b = float(x); val = stated.get(b, s["eu"][np.isclose(s["bids"], b)][0])
        if val > bv: best, bv = b, val
    return best
tf["implied_best"] = [implied_best(i, fp, st) for i, fp, st in zip(tf.scenario_id, tf.fp, tf.stated)]
tf["arith"] = [(ib == ib) and (be == be) and not np.isclose(ib, be) for ib, be in zip(tf.implied_best, tf.best_eval_bid)]
for m in MODELS:
    y = tf[(tf.model == m) & (tf.budget == 768)].reset_index(drop=True)
    gap = (y.qcalc - y.q_read).values
    wrong = ~np.isclose(y.chosen_bid.values, y.calc.values)
    A = wrong & y.arith.values
    U = wrong & ~A & ~y.complete.values
    Ur = U & y.calc_reached.values; Un = U & ~y.calc_reached.values
    I = wrong & ~A & y.complete.values & ~np.isclose(y.chosen_bid.values, y.best_eval_bid.values)
    cats = (A, Ur, Un, I)
    def shares(ids):
        T = gap[ids].sum(); return [gap[ids][c[ids]].sum() / T for c in cats]
    pt = shares(np.arange(len(y)))
    bs = np.array([shares(family_boot_ids(y.fam.values, rng)) for _ in range(2000)])
    names = ["arithmetic", "unfinished_best_reached", "unfinished_best_not_reached", "passed_over_complete"]
    R["A2"][m] = {n: dict(share=float(pt[j]), ci=[float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]) for j, n in enumerate(names)}
    R["A2"][m]["answer_not_best_evaluated_total"] = float(pt[1] + pt[3])
    R["A2"][m]["n_traces"] = {n: int(c.sum()) for n, c in zip(names, cats)}
    # A8: negative per-trace gaps
    R["A8"][m] = dict(neg_gap_traces=int((wrong & (gap < 0)).sum()), wrong_traces=int(wrong.sum()),
                      neg_gap_passed=int((I & (gap < 0)).sum()), passed=int(I.sum()))

# ---------------- A3 / A4
for m in MODELS:
    nat = tf[(tf.model == m) & (tf.budget == -1)].reset_index(drop=True)
    f768 = tf[(tf.model == m) & (tf.budget == 768)].set_index(["scenario_id", "repeat_seed"])
    res3 = {}
    for K in (5, 7, None):
        z = nat if K is None else nat[nat.K == K]
        ct = np.where(z.complete, z.comp_tok, np.inf)
        grid = np.arange(0, 1025, 8)
        F = [float((ct <= t).mean()) for t in grid]
        med = next((int(t) for t, f in zip(grid, F) if f >= 0.5), None)
        res3["all" if K is None else f"K{K}"] = dict(F768=float((ct <= 768).mean()), F1024=float((ct <= 1024).mean()), median=med,
                                                     curve=F if K is None else None)
    res3["forced768_complete"] = float(f768.complete.mean())
    # agreement: completed by 768 in the unconstrained trace vs complete in the forced 768 trace
    j = nat.set_index(["scenario_id", "repeat_seed"]).join(f768[["complete"]], rsuffix="_f")
    by768 = np.where(j.complete, j.comp_tok, np.inf) <= 768
    res3["agree_by768_vs_forced"] = float((by768 == j.complete_f.values).mean())
    R["A3"][m] = res3
    tok = nat.reasoning_tokens.fillna(0).values
    jq = nat.set_index(["scenario_id", "repeat_seed"]).join(f768[["q_read"]], rsuffix="_f")
    grp = np.where(tok <= 768, "stop<=768", np.where(tok >= 1024, "cap", "stop769-1023"))
    res4 = dict(at_cap=float((tok >= 1024).mean()), stop_le768=float((tok <= 768).mean()),
                quantiles={str(qq): float(np.quantile(tok, qq)) for qq in (0.1, 0.25, 0.5, 0.75, 0.9)})
    for gname in ("stop<=768", "stop769-1023", "cap"):
        sel = grp == gname
        if sel.sum():
            res4[gname] = dict(n=int(sel.sum()), q_uncon=float(jq.q_read.values[sel].mean()), q_forced768=float(jq.q_read_f.values[sel].mean()),
                               complete_uncon=float(nat.complete.values[sel].mean()))
    R["A4"][m] = res4

# ---------------- A5
y = tf[tf.budget.isin([768, -1]) & tf.complete].copy()
y["ignored"] = (~np.isclose(y.chosen_bid, y.best_eval_bid)).astype(int)
def pos_of(lst, b): return lst.index(b) + 1
def runner(sid):
    s = S[sid]; o = np.argsort(-s["eu"]); return float(s["bids"][o[1]])
y["best_pos_rel"] = [(pos_of(l, b) - 1) / (k - 1) for l, b, k in zip(y.order_list, y.best_eval_bid, y.K)]
y["runner_pos_rel"] = [(pos_of(l, runner(i)) - 1) / (k - 1) for l, i, k in zip(y.order_list, y.scenario_id, y.K)]
y["grp"] = y.model + "|" + y.scenario_id.astype(str) + "|" + y.budget.astype(str)
b0 = tf[tf.budget == 0].set_index(["model", "scenario_id", "repeat_seed"]).chosen_bid
for m in MODELS:
    z = y[y.model == m].reset_index(drop=True)
    var = z.groupby("grp").ignored.transform(lambda s: s.nunique() > 1)
    zz = z[var].reset_index(drop=True)
    res = dict(n_complete=int(len(z)), n_ignored=int(z.ignored.sum()), n_informative=int(len(zz)),
               n_informative_events=int(zz.ignored.sum()), n_groups=int(zz.grp.nunique()))
    fit = ConditionalLogit(zz.ignored.values, zz[["best_pos_rel", "runner_pos_rel"]].values, groups=pd.factorize(zz.grp)[0]).fit(disp=0)
    ci = np.exp(fit.conf_int())
    res.update(or_best=float(np.exp(fit.params[0])), or_best_ci=[float(ci[0, 0]), float(ci[0, 1])], p_best=float(fit.pvalues[0]))
    # budget-0 prior
    w = tf[(tf.model == m) & (tf.budget == 768)].reset_index(drop=True)
    w["b0"] = [b0.get((m, i, s_), np.nan) for i, s_ in zip(w.scenario_id, w.repeat_seed)]
    wrong = ~np.isclose(w.chosen_bid, w.calc)
    passed = w.complete & ~np.isclose(w.chosen_bid, w.best_eval_bid)
    same0 = np.isclose(w.chosen_bid, w.b0)
    res["b0_repeat_passed"] = float(same0[passed].mean()) if passed.any() else None
    res["b0_repeat_other_wrong"] = float(same0[wrong & ~passed].mean())
    res["b0_repeat_right"] = float(same0[~wrong].mean())
    res["chance_1_over_K"] = float((1 / w.K[passed]).mean()) if passed.any() else None
    R["A5"][m] = res

# ---------------- A7
st = d[d.decoding == "stochastic"].copy()
def vote(bids, probs):
    cnt = {}
    for b_, p_ in zip(bids, probs):
        c0, s0 = cnt.get(b_, (0, 0.0)); cnt[b_] = (c0 + 1, s0 + p_)
    return max(cnt.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
SIDS = sorted(S); fams = np.array([S[i]["fam"] for i in SIDS])
for m in MODELS:
    x = st[st.model == m]
    P = {b: x[x.budget == b].pivot(index="scenario_id", columns="repeat_seed", values="chosen_bid") for b in (128, 384, 768)}
    PR = {b: x[x.budget == b].pivot(index="scenario_id", columns="repeat_seed", values="constrained_option_probability") for b in (128, 384, 768)}
    seeds = list(P[128].columns)
    def qv(bud, k):
        qs = np.zeros(len(SIDS)); n = 0
        for comb in itertools.combinations(seeds, k):
            n += 1
            for j, i in enumerate(SIDS):
                qs[j] += q_of(i, vote(P[bud].loc[i, list(comb)].values, PR[bud].loc[i, list(comb)].values))
        return qs / n
    q5_128, q1_768, q2_384 = qv(128, 5), qv(768, 1), qv(384, 2)
    res = {}
    for name, (qa, qb) in {"5x128_vs_1x768": (q5_128, q1_768), "2x384_vs_1x768": (q2_384, q1_768)}.items():
        diff = qa - qb
        bs = [diff[family_boot_ids(fams, rng)].mean() for _ in range(2000)]
        res[name] = dict(q_vote=float(qa.mean()), q_single=float(qb.mean()), diff=float(diff.mean()),
                         ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    R["A7"][m] = res

json.dump(R, open(f"{OUTDIR}/t9_judge6.json", "w"), indent=1, default=float)
def r2(o): return round(float(o), 3)
for k in R:
    print("=====", k)
    for m, v in R[k].items():
        if k == "A1":
            b = {r["budget"]: r for r in v["by_budget"]}
            print(m, "cov768 st/par", r2(b[768]["stated_cov"]), r2(b[768]["parser_cov"]), "gap768 read/svany/svkm1/parser",
                  r2(b[768]["gap_read"]), r2(b[768]["gap_sv_any"]), r2(b[768]["gap_sv_km1"]), r2(b[768]["gap_parser"]),
                  "ci", {c: [r2(a) for a in x] for c, x in v["ci768"].items()}, "min diff", r2(v["sv_any_minus_read_min"]), r2(v["sv_km1_minus_read_min"]),
                  "uncon svany", r2(b[-1]["gap_sv_any"]))
        elif k == "A3":
            print(m, {kk: (vv if not isinstance(vv, dict) else {a: c for a, c in vv.items() if a != "curve"}) for kk, vv in v.items()})
        else:
            print(m, json.dumps(v, default=r2))
