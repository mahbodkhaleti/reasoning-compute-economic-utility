"""Batch 6: trace analyses redone with parser v2 (validated against 150 hand-labeled traces).
An option counts as evaluated when a number within 2% of its true expected profit appears in the trace.
"""
import json, numpy as np, pandas as pd
import statsmodels.formula.api as smf
exec(open("batch4_judge_r1.py").read().split("# ================= F1")[0])
rng = np.random.default_rng(20261110)

def km(time, event, grid):
    order = np.argsort(time); time, event = np.asarray(time, float)[order], np.asarray(event)[order]
    n = len(time); surv = 1.0; pts = []; at_risk = n; i = 0
    while i < n:
        tt = time[i]; dd = 0; cc = 0
        while i < n and time[i] == tt:
            dd += event[i]; cc += 1 - event[i]; i += 1
        if at_risk > 0 and dd > 0: surv *= (1 - dd / at_risk)
        pts.append((tt, 1 - surv)); at_risk -= dd + cc
    pts = np.array(pts)
    return np.array([pts[pts[:, 0] <= gg, 1].max() if (pts[:, 0] <= gg).any() else 0.0 for gg in grid])
def qfrom(grid, cdf, qq):
    idx = np.where(cdf >= qq)[0]; return float(grid[idx[0]]) if len(idx) else None

tf = pd.read_csv("trace_first_v2.csv")
tf["K"] = tf.scenario_id.map(lambda i: S[i]["K"])
tf["fp"] = tf.first_pos.map(json.loads)
tf["n_eval"] = tf.fp.map(len)
tf["complete"] = tf.n_eval == tf.K
tok = tf.reasoning_tokens.fillna(0).values; nch = np.maximum(tf.n_chars.values, 1)
tf["comp_tok"] = [max(fp.values()) / n * t if c else np.nan for fp, n, t, c in zip(tf.fp, nch, tok, tf.complete)]
tf["ev256"] = [sum(1 for p in fp.values() if p / n * t <= 256) for fp, n, t in zip(tf.fp, nch, tok)]
tf["ev128"] = [sum(1 for p in fp.values() if p / n * t <= 128) for fp, n, t in zip(tf.fp, nch, tok)]
tf["q_read"] = [q_of(i, b) for i, b in zip(tf.scenario_id, tf.chosen_bid)]
tf["calc"] = tf.scenario_id.map(lambda i: S[i]["calc"])
tf["qcalc"] = tf.scenario_id.map(lambda i: q_of(i, S[i]["calc"]))
# best evaluated option so far (true values of evaluated options) = what a correct comparison of the stated options gives
tf["q_best_evaluated"] = [max(q_of(i, float(b)) for b in fp) if len(fp) else np.nan for i, fp in zip(tf.scenario_id, tf.fp)]
R = {}
grid = np.arange(0, 1025, 8)
for m in MODELS:
    x = tf[tf.model == m]; res = {}
    for bud in [256, 512, 768, -1]:
        y = x[x.budget == bud]
        res[bud] = dict(complete=float(y.complete.mean()), q=float(y.q_read.mean()),
                        q_complete=float(y[y.complete].q_read.mean()) if y.complete.any() else None,
                        q_unfinished=float(y[~y.complete].q_read.mean()) if (~y.complete).any() else None,
                        match_calc_complete=float(np.isclose(y[y.complete].chosen_bid, y[y.complete].calc).mean()) if y.complete.any() else None,
                        n_eval=float(y.n_eval.mean()))
    nat = x[x.budget == -1]
    time = np.where(nat.complete, nat.comp_tok, nat.reasoning_tokens); ev = nat.complete.values.astype(int)
    for kk, sub in [("all", np.ones(len(nat), bool)), ("K5", (nat.K == 5).values), ("K7", (nat.K == 7).values)]:
        cdf = km(time[sub], ev[sub], grid)
        res["km_" + kk] = dict(p256=float(cdf[grid == 256][0]), p512=float(cdf[grid == 512][0]), p768=float(cdf[grid == 768][0]),
                               p1024=float(cdf[-1]), median=qfrom(grid, cdf, 0.5), p90=qfrom(grid, cdf, 0.9))
    # natural traces that stopped before the cap
    fin = nat[nat.reasoning_tokens < 1024]
    res["natural"] = dict(at_cap=float((nat.reasoning_tokens >= 1024).mean()), q_finished=float(fin.q_read.mean()) if len(fin) else None,
                          n_finished=int(len(fin)), q_all=float(nat.q_read.mean()))
    # prefix consistency: completion at 768 implied by the natural trace vs observed in the forced-768 trace
    f7 = x[x.budget == 768].set_index(["scenario_id", "repeat_seed"]).complete
    pred = (nat.set_index(["scenario_id", "repeat_seed"]).comp_tok <= 768).reindex(f7.index).fillna(False)
    res["prefix_agree_768"] = float((pred.values == f7.values).mean())
    # loss decomposition at 768 (utility ratio, relative to calculator)
    y = x[x.budget == 768].reset_index(drop=True)
    short = (y.qcalc - y.q_read).values; mis = ~np.isclose(y.chosen_bid, y.calc)
    inc = (mis & ~y.complete).values; cmp_ = (mis & y.complete).values
    fams = y.scenario_id.map(lambda i: S[i]["fam"]).values; idx = {f: np.where(fams == f)[0] for f in FAM}
    bs = []
    for _ in range(2000):
        ids = np.concatenate([idx[f] for p in PRODUCTS for f in rng.choice([f for f in FAM if f.startswith(p)], 18, replace=True)])
        bs.append(short[ids][inc[ids]].sum() / short[ids].sum())
    res["decomp"] = dict(unfinished=float(short[inc].sum() / short.sum()), complete_wrong=float(short[cmp_].sum() / short.sum()),
                         unfinished_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    # unfinished traces: does the readout pick the best of the options evaluated so far?
    u = y[~y.complete]
    res["unfinished_768"] = dict(q_read=float(u.q_read.mean()), q_best_evaluated=float(u.q_best_evaluated.mean()), n_eval=float(u.n_eval.mean()))
    R[m] = res

# selection: within-scenario fixed effects; calculator q on scenarios that complete vs not; completion logit
Z = tf[tf.budget.isin([512, 768, -1])].copy(); Z["cell"] = Z.model + "|" + Z.scenario_id.astype(str); Z["ci"] = Z.complete.astype(int)
fe = smf.ols("q_read ~ ci + C(budget) + C(cell)", data=Z).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(Z.scenario_id)[0]})
raw = Z.groupby("complete").q_read.mean()
z7 = tf[tf.budget == 768]; scc = z7.groupby("scenario_id").complete.mean()
z7 = z7.assign(gap12=z7.scenario_id.map(lambda i: S[i]["gap12"]),
               v_rel=z7.scenario_id.map(lambda i: S[i]["v"] / np.median([S[j]["v"] for j in S if S[j]["product"] == S[i]["product"]])),
               product=z7.scenario_id.map(lambda i: S[i]["product"]), comp=z7.complete.astype(int))
lg = smf.logit("comp ~ C(K) + gap12 + v_rel + C(product) + C(model)", data=z7).fit(disp=0)
R["selection"] = dict(fe_coef=float(fe.params["ci"]), fe_se=float(fe.bse["ci"]), raw_diff=float(raw[True] - raw[False]),
                      qcalc_high_completion=float(np.mean([q_of(i, S[i]["calc"]) for i in scc.index[scc >= scc.median()]])),
                      qcalc_low_completion=float(np.mean([q_of(i, S[i]["calc"]) for i in scc.index[scc < scc.median()]])),
                      logit={k: dict(coef=float(lg.params[k]), p=float(lg.pvalues[k])) for k in lg.params.index if not k.startswith("C(model)")})

# routing by options evaluated within the first 128/256 tokens (probe charged), scored on the ratio gain
g = pd.read_csv("saved_records.csv.gz"); g = g[g.decoding == "greedy"]
gq = g[g.budget.isin([0, 128, 256, 768])].pivot_table(index=["model", "scenario_id", "repeat_seed"], columns="budget", values="test_utility_ratio").reset_index()
natidx = tf[tf.budget == -1].set_index(["model", "scenario_id", "repeat_seed"])
rout = {}
for m in MODELS:
    x = gq[gq.model == m]; key = list(zip([m] * len(x), x.scenario_id, x.repeat_seed))
    res = {}
    for nm, col, pb in [("options_by_128", "ev128", 128), ("options_by_256", "ev256", 256)]:
        e = natidx[col].reindex(key).values.astype(float); jit = rng.uniform(0, 1e-6, len(x))
        for direction, sgn in [("fewer_first", -1), ("more_first", 1)]:
            order = np.argsort(-(sgn * e + jit), kind="stable"); n = len(x); gr = np.linspace(0, 1, 101)
            qprobe = x[pb].values
            qpol = np.array([np.where(np.isin(np.arange(n), order[:int(round(s_ * n))]), x[768].values, qprobe).mean() for s_ in gr])
            kept = (qpol - qprobe.mean()) / (x[768].mean() - qprobe.mean())
            res[f"{nm}_{direction}"] = float(np.trapezoid(kept, gr))
    rout[m] = res
R["routing_progress"] = rout
json.dump(R, open("out/batch6_traces_v2.json", "w"), indent=1, default=float)
for m in MODELS:
    r = R[m]
    print(LAB[m], {b: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r[b].items()} for b in [512, 768, -1]})
    print("   km", {k: r[k] for k in ["km_all", "km_K5", "km_K7"]})
    print("   nat", r["natural"], "prefix", round(r["prefix_agree_768"], 3), "decomp", r["decomp"], "unfin", r["unfinished_768"])
    print("   routing", {k: round(v, 3) for k, v in R["routing_progress"][m].items()})
print(R["selection"])
