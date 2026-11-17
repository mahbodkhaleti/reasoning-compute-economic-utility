"""Batch 5: CPU analyses requested by blind review round 2.
F1 tokens-to-complete (Kaplan-Meier from the natural run, censored), q conditional on completion, prefix check
F2 tokens per option evaluated; salvage readout for unfinished traces
F3 eBay vs synthetic: completion and tokens per option on synthetic traces; eBay scored in-sample; scenario features
F4 complete vs unfinished within scenario (scenario fixed effects); calculator q on completed vs not; completion logit
F6 routing by options evaluated within the first 256 tokens (probe charged), and by K
F8 dollar loss vs the calculator by budget
F9 bootstrap intervals for Table 2 shares
"""
import json, numpy as np, pandas as pd
import statsmodels.formula.api as smf
exec(open("batch4_judge_r1.py").read().split("# ================= F1")[0])
rng = np.random.default_rng(20261109)
R = {}

def kept_matches(row, disp):
    """filtered matches: products whose probability equals the displayed probability (2% rel), plus label summaries"""
    out = []
    for b, p, r, pos, kind in sorted(row, key=lambda z: z[3]):
        b = float(b)
        if b not in disp: continue
        if kind == "prod" and (p is None or abs(p - disp[b]) > 0.02 * disp[b] + 1e-12): continue
        out.append((b, r, pos))
    return out

t = pd.read_csv("trace_matches.csv")
t["K"] = t.scenario_id.map(lambda i: S[i]["K"])
rows = []
for r in t.itertuples():
    s = S[r.scenario_id]; disp = dict(zip(s["bids"], s["dp"]))
    km = kept_matches(json.loads(r.values), disp)
    first, last = {}, {}
    for b, val, pos in km:
        first.setdefault(b, pos); last[b] = val
    nch = max(r.n_chars, 1); tok = r.reasoning_tokens if r.reasoning_tokens == r.reasoning_tokens else 0
    to_tok = lambda pos: pos / nch * tok
    complete = len(last) == r.K
    comp_tok = to_tok(max(first.values())) if complete else np.nan
    firsts_tok = sorted(to_tok(p) for p in first.values())
    ev256 = sum(1 for p in first.values() if to_tok(p) <= 256)
    impl = max(last.items(), key=lambda kv: kv[1])[0] if complete else np.nan
    salv = max(last.items(), key=lambda kv: kv[1])[0] if (not complete and len(last)) else np.nan
    rows.append(dict(model=r.model, scenario_id=r.scenario_id, order=r.repeat_seed, budget=r.budget, K=r.K, tokens=tok,
                     complete=complete, n_stated=len(last), comp_tok=comp_tok, firsts=firsts_tok, ev256=ev256,
                     chosen=r.chosen_bid, implied=impl, salvage=salv, close=r.close))
T = pd.DataFrame(rows)
T["q_read"] = [q_of(i, b) for i, b in zip(T.scenario_id, T.chosen)]
T["q_impl"] = [q_of(i, b) if b == b else np.nan for i, b in zip(T.scenario_id, T.implied)]
T["q_salv"] = [q_of(i, b) if b == b else np.nan for i, b in zip(T.scenario_id, T.salvage)]
T["qcalc"] = T.scenario_id.map(lambda i: q_of(i, S[i]["calc"]))

def km(time, event, grid):
    """Kaplan-Meier P(complete by t)."""
    order = np.argsort(time); time, event = np.asarray(time)[order], np.asarray(event)[order]
    n = len(time); surv = 1.0; pts = []; at_risk = n
    i = 0
    while i < n:
        tt = time[i]; d = 0; c = 0
        while i < n and time[i] == tt:
            d += event[i]; c += 1 - event[i]; i += 1
        if at_risk > 0 and d > 0: surv *= (1 - d / at_risk)
        pts.append((tt, 1 - surv)); at_risk -= d + c
    pts = np.array(pts)
    return np.array([pts[pts[:, 0] <= gg, 1].max() if (pts[:, 0] <= gg).any() else 0.0 for gg in grid])
def quantile_from(grid, cdf, qq):
    idx = np.where(cdf >= qq)[0]; return float(grid[idx[0]]) if len(idx) else None

F1 = {}; grid = np.arange(0, 1025, 8)
for m in MODELS:
    nat = T[(T.model == m) & (T.budget == -1)]
    time = np.where(nat.complete, nat.comp_tok, nat.tokens); event = nat.complete.values.astype(int)
    res = {}
    for kk, sub in [("all", np.ones(len(nat), bool)), ("K5", (nat.K == 5).values), ("K7", (nat.K == 7).values)]:
        cdf = km(time[sub], event[sub], grid)
        res[kk] = dict(p_by_512=float(cdf[grid == 512][0]), p_by_768=float(cdf[grid == 768][0]), p_by_1024=float(cdf[-1]),
                       median=quantile_from(grid, cdf, 0.5), p90=quantile_from(grid, cdf, 0.9))
    # prefix check: completion by 768 predicted from the natural trace vs observed in the forced-768 trace
    f768 = T[(T.model == m) & (T.budget == 768)].set_index(["scenario_id", "order"]).complete
    pred = (nat.set_index(["scenario_id", "order"]).comp_tok <= 768).reindex(f768.index)
    res["prefix_agreement_768"] = float((pred.fillna(False).values == f768.values).mean())
    # q conditional on completion
    res["q_given"] = {}
    for bud in [512, 768, -1]:
        y = T[(T.model == m) & (T.budget == bud)]
        res["q_given"][bud] = dict(complete=float(y.complete.mean()), q_complete=float(y[y.complete].q_read.mean()) if y.complete.any() else None,
                                   q_unfinished=float(y[~y.complete].q_read.mean()) if (~y.complete).any() else None,
                                   q_impl_complete=float(y[y.complete].q_impl.mean()) if y.complete.any() else None)
    F1[m] = res
R["F1"] = F1

# ================= F2 tokens per option; salvage
F2 = {}
for m in MODELS:
    y = T[(T.model == m) & (T.budget == -1) & T.complete]
    pts = [(k + 1, f) for fl in y.firsts for k, f in enumerate(fl)]
    if len(pts) > 10:
        a = np.array(pts); slope, icpt = np.polyfit(a[:, 0], a[:, 1], 1)
    else:
        slope = icpt = np.nan
    u = T[(T.model == m) & (T.budget == 768) & ~T.complete]
    F2[m] = dict(tokens_per_option=float(slope), intercept=float(icpt),
                 needed_K5=float(icpt + 5 * slope), needed_K7=float(icpt + 7 * slope),
                 salvage_q=float(u.q_salv.mean()), readout_q_unfinished=float(u.q_read.mean()),
                 salvage_share=float(u.salvage.notna().mean()), n_stated_unfinished=float(u.n_stated.mean()))
R["F2"] = F2

# ================= F3 eBay vs synthetic
syn = pd.read_csv("synthetic_matches.csv")
F3 = {}
for m in MODELS:
    x = syn[syn.model == m]; res = {}
    for bud in [512, 768]:
        y = x[x.budget == bud]; comp = []
        for r in y.itertuples():
            disp = {b: p for b, p in json.loads(r.displayed)}; ms = json.loads(r.matches); st = set()
            for b, p, val, pos in ms:
                if b in disp and abs(p - disp[b]) <= 0.02 * disp[b] + 1e-12: st.add(b)
            comp.append(len(st) == r.K)
        comp = np.array(comp)
        res[bud] = dict(complete=float(comp.mean()), q=float(y.utility_ratio.mean()),
                        q_complete=float(y.utility_ratio.values[comp].mean()) if comp.any() else None,
                        q_unfinished=float(y.utility_ratio.values[~comp].mean()) if (~comp).any() else None)
    res["K_share7"] = float((x.K == 7).mean()); res["v_range"] = [float(x.v.min()), float(x.v.max())]
    F3[m] = res
# eBay scored in-sample (construction probabilities): gap recovered relative to calculator (=1 in-sample)
d = pd.read_csv("saved_records.csv.gz"); g = d[d.decoding == "greedy"].copy()
def q_in(i, b):
    s = S[i]; u = s["dp"] * (s["v"] - s["bids"]); return u[np.isclose(s["bids"], b)][0] / u.max()
g["q_in"] = [q_in(i, b) for i, b in zip(g.scenario_id, g.chosen_bid)]
ins = {}
for m in MODELS:
    z = g[g.model == m].groupby("budget").q_in.mean()
    ins[m] = dict(q0=float(z[0]), q768=float(z[768]), gap_recovered=float((z[768] - z[0]) / (1 - z[0])))
F3["ebay_in_sample"] = ins
F3["ebay_features"] = dict(K_share7=float(np.mean([S[i]["K"] == 7 for i in S])),
                           gap12_median=float(np.median([S[i]["gap12"] for i in S])))
synS = syn[syn.budget == 0].drop_duplicates("scenario_id")
gaps = []
for r in synS.itertuples():
    disp = json.loads(r.displayed); eu = sorted([p * (r.v - b) for b, p in disp], reverse=True); gaps.append((eu[0] - eu[1]) / eu[0])
F3["synthetic_features"] = dict(gap12_median=float(np.median(gaps)), K_share7=float((synS.K == 7).mean()))
R["F3"] = F3

# ================= F4 selection: complete vs unfinished within scenario
F4 = {}
Z = T[T.budget.isin([512, 768, -1])].copy(); Z["cell"] = Z.model + "|" + Z.scenario_id.astype(str)
Z["complete_i"] = Z.complete.astype(int)
mod = smf.ols("q_read ~ complete_i + C(budget) + C(cell)", data=Z).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(Z.scenario_id)[0]})
F4["fe_complete_coef"] = dict(coef=float(mod.params["complete_i"]), se=float(mod.bse["complete_i"]), p=float(mod.pvalues["complete_i"]))
raw = Z.groupby("complete").q_read.mean(); F4["raw_diff"] = float(raw[True] - raw[False])
# calculator q on scenarios where 768 traces complete vs not
z7 = T[T.budget == 768]
sc_comp = z7.groupby("scenario_id").complete.mean()
F4["qcalc_scenarios_any_complete"] = float(np.mean([q_of(i, S[i]["calc"]) for i in sc_comp.index[sc_comp > 0]]))
F4["qcalc_scenarios_none_complete"] = float(np.mean([q_of(i, S[i]["calc"]) for i in sc_comp.index[sc_comp == 0]]))
z7 = z7.assign(gap12=z7.scenario_id.map(lambda i: S[i]["gap12"]), v_rel=z7.scenario_id.map(lambda i: S[i]["v"] / np.median([S[j]["v"] for j in S if S[j]["product"] == S[i]["product"]])),
               product=z7.scenario_id.map(lambda i: S[i]["product"]), comp=z7.complete.astype(int))
lg = smf.logit("comp ~ C(K) + gap12 + v_rel + C(product) + C(model)", data=z7).fit(disp=0)
F4["logit"] = {k: dict(coef=float(lg.params[k]), p=float(lg.pvalues[k])) for k in lg.params.index if not k.startswith("C(model)")}
R["F4"] = F4

# ================= F6 routing by options evaluated within the first 256 tokens (probe = 256 tokens) and by K
F6 = {}
gq = g[g.budget.isin([0, 256, 768])].pivot_table(index=["model", "scenario_id", "repeat_seed"], columns="budget", values="test_utility_ratio").reset_index()
nat = T[T.budget == -1].set_index(["model", "scenario_id", "order"])
for m in MODELS:
    x = gq[gq.model == m].copy()
    key = list(zip([m] * len(x), x.scenario_id, x.repeat_seed))
    ev = nat.ev256.reindex(key).values.astype(float)
    Kv = x.scenario_id.map(lambda i: S[i]["K"]).values.astype(float)
    jit = rng.uniform(0, 1e-6, len(x))
    res = {}
    for name, score, qprobe, cost in [("few_options_by_256", -ev, x[256].values, 256), ("many_options_by_256", ev, x[256].values, 256),
                                      ("K", Kv, x[0].values, 0)]:
        order = np.argsort(-(score + jit), kind="stable"); n = len(x); gridq = np.linspace(0, 1, 101)
        qpol = np.array([np.where(np.isin(np.arange(n), order[:int(round(s_ * n))]), x[768].values, qprobe).mean() for s_ in gridq])
        kept = (qpol - qprobe.mean()) / (x[768].mean() - qprobe.mean())
        res[name] = dict(area=float(np.trapezoid(kept, gridq)), probe=cost)
    F6[m] = res
R["F6"] = F6

# ================= F8 dollar loss vs calculator by budget
F8 = {}
for m in MODELS:
    y = g[g.model == m]; calc_u = y.scenario_id.map(lambda i: S[i]["tu"][np.isclose(S[i]["bids"], S[i]["calc"])][0])
    F8[m] = {int(b): float((calc_u[y.budget == b] - y.test_utility[y.budget == b]).mean()) for b in [0, 256, 512, 768, -1]}
R["F8"] = F8

# ================= F9 bootstrap intervals for Table 2 loss shares
F9 = {}
for m in MODELS:
    y = T[(T.model == m) & (T.budget == 768)].reset_index(drop=True)
    short = (y.qcalc - y.q_read).values; mis = ~np.isclose(y.chosen, y.scenario_id.map(lambda i: S[i]["calc"]))
    inc = (mis & ~y.complete).values
    def share(ids): return short[ids][inc[ids]].sum() / short[ids].sum()
    fams = y.scenario_id.map(lambda i: S[i]["fam"]).values; idx = {f: np.where(fams == f)[0] for f in FAM}
    bs = []
    for _ in range(2000):
        ids = np.concatenate([idx[f] for p in PRODUCTS for f in rng.choice([f for f in FAM if f.startswith(p)], 18, replace=True)])
        bs.append(share(ids))
    F9[m] = dict(unfinished_share=float(share(np.arange(len(y)))), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                 complete_ci=fam_boot(y, lambda z: z.complete.mean(), n=1000))
R["F9"] = F9
json.dump(R, open("out/batch5_results.json", "w"), indent=1, default=float)
print("done")
