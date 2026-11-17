"""Batch 4: CPU analyses requested by blind review round 1.
F1 trace-implied bids, coverage by budget, calculator-gap decomposition, natural-length run vs tokens used
F3 synthetic IPV replication
F4 cost-charged cascades with early-budget probes and order agreement, scored on ratio gain
F5 majority-vote frontier (k = 1..5 at 128/384/768)
F6 per-product ratio gains with intervals; regret regression; cross-order agreement by budget
F7 latency and energy per decision
"""
import json, itertools, os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

OUT = "out"; rng = np.random.default_rng(20261108)
MODELS = ["deepseek_r1_llama_8b", "phi4_mini_reasoning", "qwen3_4b", "qwen3_8b", "qwen3_14b", "granite_3_2_8b_instruct"]
LAB = {"deepseek_r1_llama_8b": "DeepSeek-8B", "phi4_mini_reasoning": "Phi-mini", "qwen3_4b": "Qwen3-4B",
       "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B", "granite_3_2_8b_instruct": "Granite-8B"}
R = {}
sc = pd.read_csv("scenarios.csv")
S = {}
for r in sc.itertuples():
    c = json.loads(r.candidates_json)
    bids = np.array([x["bid"] for x in c], float)
    S[r.scenario_id] = dict(product=r.category, v=float(r.value), K=r.option_count, bids=bids,
                            dp=np.round(np.array([x["train_p"] for x in c]), 4),
                            tu=np.array([x["test_u"] for x in c]), fam=f"{r.category}|{r.value}")
for s in S.values():
    s["calc"] = s["bids"][np.argmax(s["dp"] * (s["v"] - s["bids"]))]
    s["ustar"] = s["tu"].max()
    eu = np.sort(s["dp"] * (s["v"] - s["bids"]))[::-1]; s["gap12"] = (eu[0] - eu[1]) / eu[0]
FAM = sorted({s["fam"] for s in S.values()}); PRODUCTS = sorted({s["product"] for s in S.values()})
def q_of(sid, b):
    s = S[sid]; return s["tu"][np.isclose(s["bids"], b)][0] / s["ustar"]
def fam_boot(df, stat, n=2000, key="scenario_id"):
    fams = df[key].map(lambda i: S[i]["fam"]).values
    idx = {f: np.where(fams == f)[0] for f in FAM}
    out = []
    for _ in range(n):
        ids = np.concatenate([idx[f] for p in PRODUCTS for f in rng.choice([f for f in FAM if f.startswith(p)], 18, replace=True)])
        out.append(stat(df.iloc[ids]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

# ================= F1
t = pd.read_csv("trace_values.csv")
t["vals"] = t["values"].map(json.loads); t["ev"] = t["evaluated"].map(json.loads)
t["K"] = t.scenario_id.map(lambda i: S[i]["K"])
t["complete"] = [len(v) == k for v, k in zip(t.vals, t.K)]
t["covered"] = [len(e) == k for e, k in zip(t.ev, t.K)]
def implied(row):
    if not row.complete: return np.nan
    items = sorted(row.vals.items(), key=lambda kv: -kv[1]); return float(items[0][0])
t["implied"] = t.apply(implied, axis=1)
t["calc"] = t.scenario_id.map(lambda i: S[i]["calc"])
t["q_read"] = [q_of(i, b) for i, b in zip(t.scenario_id, t.chosen_bid)]
t["q_impl"] = [q_of(i, b) if not np.isnan(b) else np.nan for i, b in zip(t.scenario_id, t.implied)]
t["read_is_calc"] = np.isclose(t.chosen_bid, t.calc)
F1 = {}
for m in MODELS:
    x = t[t.model == m]; res = {"by_budget": {}}
    for bud in [128, 256, 512, 768, -1]:
        y = x[x.budget == bud]
        res["by_budget"][bud] = dict(covered=float(y.covered.mean()), complete=float(y.complete.mean()),
                                     q_read=float(y.q_read.mean()),
                                     q_read_complete=float(y[y.complete].q_read.mean()) if y.complete.any() else None,
                                     q_impl_complete=float(y[y.complete].q_impl.mean()) if y.complete.any() else None,
                                     impl_is_calc=float(np.isclose(y[y.complete].implied, y[y.complete].calc).mean()) if y.complete.any() else None,
                                     read_eq_impl=float(np.isclose(y[y.complete].implied, y[y.complete].chosen_bid).mean()) if y.complete.any() else None,
                                     tokens=float(y.reasoning_tokens.mean()), close=float(y.close.mean()))
    y = x[x.budget == 768]; mis = y[~y.read_is_calc]
    inc = ~mis.complete
    wrong_in_trace = mis.complete & ~np.isclose(mis.implied, mis.calc)
    readout = mis.complete & np.isclose(mis.implied, mis.calc)
    # loss decomposition in utility-ratio units: share of the total q-shortfall vs calculator
    qcalc = y.scenario_id.map(lambda i: q_of(i, S[i]["calc"]))
    short = (qcalc - y.q_read)
    tot = short.sum()
    res["decomp_768"] = dict(mismatch_share=float((~y.read_is_calc).mean()),
                             incomplete=float(inc.mean()), in_trace_error=float(wrong_in_trace.mean()), readout=float(readout.mean()),
                             loss_incomplete=float(short[~y.read_is_calc][inc.values].sum() / tot),
                             loss_in_trace=float(short[~y.read_is_calc][wrong_in_trace.values].sum() / tot),
                             loss_readout=float(short[~y.read_is_calc][readout.values].sum() / tot))
    # natural-length run: q vs tokens actually used; matched comparison with forced 768
    nat = x[x.budget == -1].copy()
    nat["bin"] = pd.cut(nat.reasoning_tokens, [-1, 511, 767, 1023, 1024], labels=["<512", "512-767", "768-1023", "cap"])
    res["natural"] = dict(mean_tokens=float(nat.reasoning_tokens.mean()), at_cap=float((nat.reasoning_tokens >= 1024).mean()),
                          close=float(nat.close.mean()), q=float(nat.q_read.mean()),
                          q_by_bin={str(k): (float(v.q_read.mean()), int(len(v))) for k, v in nat.groupby("bin", observed=False)},
                          q_finished=float(nat[nat.reasoning_tokens < 1024].q_read.mean()) if (nat.reasoning_tokens < 1024).any() else None,
                          n_finished=int((nat.reasoning_tokens < 1024).sum()))
    mt = nat[(nat.reasoning_tokens >= 640) & (nat.reasoning_tokens <= 896)]
    if len(mt):
        f7 = x[x.budget == 768].set_index(["scenario_id", "repeat_seed"]).q_read
        pair = mt.set_index(["scenario_id", "repeat_seed"]).q_read
        common = pair.index.intersection(f7.index)
        res["matched_640_896"] = dict(n=int(len(common)), q_natural=float(pair.loc[common].mean()), q_forced768=float(f7.loc[common].mean()))
    F1[m] = res
R["F1"] = F1

# ================= F3 synthetic
syn = pd.read_csv("synthetic_summary.csv")
F3 = {}
for m in MODELS:
    x = syn[syn.model == m]; qb = x.groupby("budget").utility_ratio.mean()
    F3[m] = dict(q=qb.round(4).to_dict(), dq=float(qb[768] - qb[0]), gap_closed=float((qb[768] - qb[0]) / (1 - qb[0])),
                 exact_768=float(x[x.budget == 768].exact.mean()), tokens_768=float(x[x.budget == 768].realized_tokens.mean()))
R["F3"] = F3

# ================= F4 cascades (ratio gain)
d = pd.read_csv("saved_records.csv.gz"); g = d[d.decoding == "greedy"].copy()
g["q"] = g.test_utility_ratio
F4 = {}
for m in MODELS:
    x = g[g.model == m]
    P = x.pivot_table(index=["scenario_id", "repeat_seed"], columns="budget", values=["q", "chosen_bid", "constrained_option_probability"])
    q0, q64, q128, q512, q768 = (P[("q", b)].values for b in [0, 64, 128, 512, 768])
    b0, b64, b128 = (P[("chosen_bid", b)].values for b in [0, 64, 128])
    p64, p128 = P[("constrained_option_probability", 64)].values, P[("constrained_option_probability", 128)].values
    sids = P.index.get_level_values(0)
    # order disagreement at budget 0: number of distinct bids across the 3 orders
    nd = x[x.budget == 0].groupby("scenario_id").chosen_bid.nunique()
    dis0 = sids.map(nd).values.astype(float)
    gap12 = np.array([S[i]["gap12"] for i in sids])
    jit = rng.uniform(0, 1e-6, len(q0))
    probes = {
        "agree_0_64": (64, (b0 != b64).astype(float), q64),
        "conf_64": (64, -p64, q64),
        "conf_128": (128, -p128, q128),
        "order_disagree_0": (0, dis0, q0),
        "top2_gap_calc": (0, -gap12, q0),
        "random_0": (0, rng.uniform(size=len(q0)), q0),
    }
    res = {}
    for name, (cost, score, qprobe) in probes.items():
        order = np.argsort(-(score + jit), kind="stable"); n = len(q0)
        grid = np.linspace(0, 1, 101); qpol = []
        for s_ in grid:
            k = int(round(s_ * n)); esc = np.zeros(n, bool); esc[order[:k]] = True
            qpol.append(np.where(esc, q768, qprobe).mean())
        qpol = np.array(qpol)
        kept = (qpol - qprobe.mean()) / (q768.mean() - qprobe.mean())
        # match the average spend of a fixed 512-token budget: cost + s*768 = 512
        s512 = (512 - cost) / 768
        q_at_512 = float(np.interp(s512, grid, qpol))
        res[name] = dict(probe_tokens=cost, area=float(np.trapezoid(kept, grid)), q_at_512_spend=q_at_512,
                         fixed_512=float(q512.mean()), q768=float(q768.mean()), qprobe=float(qprobe.mean()))
    F4[m] = res
R["F4"] = F4

# ================= F5 voting frontier
st = d[d.decoding == "stochastic"]
F5 = {}
tu = {i: (S[i]["bids"], S[i]["tu"], S[i]["ustar"]) for i in S}
for m in MODELS:
    x = st[st.model == m]; pts = []
    for bud in [128, 384, 768]:
        Pb = x[x.budget == bud].pivot(index="scenario_id", columns="repeat_seed", values="chosen_bid")
        Pp = x[x.budget == bud].pivot(index="scenario_id", columns="repeat_seed", values="constrained_option_probability")
        seeds = list(Pb.columns)
        for k in range(1, 6):
            qs = []
            for comb in itertools.combinations(seeds, k):
                for i in Pb.index:
                    cnt = {}
                    for b_, p_ in zip(Pb.loc[i, list(comb)], Pp.loc[i, list(comb)]):
                        c0, s0 = cnt.get(b_, (0, 0.0)); cnt[b_] = (c0 + 1, s0 + p_)
                    w = max(cnt.items(), key=lambda kv: (kv[1][0], kv[1][1] / kv[1][0]))[0]
                    qs.append(q_of(i, w))
            pts.append(dict(budget=bud, k=k, tokens=bud * k, q=float(np.mean(qs))))
    P_ = sorted(pts, key=lambda r: (r["tokens"], -r["q"])); best = -1; front = []
    for r in P_:
        if r["q"] > best: front.append((r["budget"], r["k"], r["tokens"], round(r["q"], 3))); best = r["q"]
    F5[m] = dict(points=pts, frontier=front)
R["F5"] = F5

# ================= F6 heterogeneity
F6 = {}
cell = g[g.budget.isin([0, 768])].groupby(["model", "scenario_id", "budget"]).q.mean().unstack().reset_index()
cell["dq"] = cell[768] - cell[0]
for m in MODELS:
    c = cell[cell.model == m].reset_index(drop=True)
    c["product"] = c.scenario_id.map(lambda i: S[i]["product"])
    F6[m] = {p: dict(dq=float(c[c["product"] == p].dq.mean()),
                     ci=fam_boot(c[c["product"] == p].reset_index(drop=True), lambda z: z.dq.mean(), n=1000)) for p in PRODUCTS}
# regression of 768 shortfall (1 - q) on scenario features, pooled over models with model fixed effects, clustered by family
g768 = g[g.budget == 768].copy()
g768["short"] = 1 - g768.q
g768["K"] = g768.scenario_id.map(lambda i: S[i]["K"]); g768["gap12"] = g768.scenario_id.map(lambda i: S[i]["gap12"])
g768["v_rel"] = g768.scenario_id.map(lambda i: S[i]["v"] / np.median([S[j]["v"] for j in S if S[j]["product"] == S[i]["product"]]))
g768["fam"] = g768.scenario_id.map(lambda i: S[i]["fam"])
g768["product"] = g768.scenario_id.map(lambda i: S[i]["product"])
mod = smf.ols("short ~ C(model) + C(product) + C(K) + gap12 + v_rel", data=g768).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g768.fam)[0]})
F6["regression"] = {k: dict(coef=float(mod.params[k]), p=float(mod.pvalues[k])) for k in mod.params.index if not k.startswith("C(model)")}
F6["regression_r2"] = float(mod.rsquared)
# cross-order agreement by budget
agr = {}
for m in MODELS:
    x = g[g.model == m]
    agr[m] = {int(b): float((x[x.budget == b].groupby("scenario_id").chosen_bid.nunique() == 1).mean()) for b in [0, 64, 128, 256, 512, 768, -1]}
F6["order_agreement"] = agr
R["F6"] = F6

# ================= F7 latency and energy at 768
F7 = {}
for m in MODELS:
    y = g[(g.model == m) & (g.budget == 768)]
    sec = y.elapsed_seconds + y.constrained_readout_elapsed_seconds
    j = y.energy_joules + y.constrained_readout_energy_joules
    F7[m] = dict(latency_s=float(sec.mean()), readout_s=float(y.constrained_readout_elapsed_seconds.mean()),
                 energy_wh=float(j.mean() / 3600), energy_cost_usd_at_0p15=float(j.mean() / 3.6e6 * 0.15))
R["F7"] = F7
json.dump(R, open(f"{OUT}/batch4_results.json", "w"), indent=1, default=float)
print("done")
