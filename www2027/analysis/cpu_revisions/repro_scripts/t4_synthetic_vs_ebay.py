"""Task 4: why does the synthetic replication recover more of the gap than eBay?
Compares scenario features (digits of v and bids, displayed-probability precision, spread of expected profits,
top-two margin, K) and trace completion (parser v2, 2%; synthetic absolute floor 1e-4 because payoffs are small),
then regresses per-decision gain (q at 768 minus q at 0) on source and features with model fixed effects,
SEs clustered by scenario. eBay gain is computed both on held-out probabilities and on the displayed probabilities
(in-sample), since synthetic payoffs are exact and equal to the displayed ones.
"""
import json, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from common_rev import *

def digits(x): return len(str(int(abs(x)))) if x >= 1 else 1
# ---- eBay decisions
d = records(); g = d[d.decoding == "greedy"].copy()
def q_in(i, b):
    s = S[i]; return float(s["eu"][np.isclose(s["bids"], b)][0] / s["eu"].max())
g["q_in"] = [q_in(i, b) for i, b in zip(g.scenario_id, g.chosen_bid)]
_p = g[g.budget.isin([0, 768])].pivot_table(index=["model", "scenario_id", "repeat_seed"], columns="budget", values=["test_utility_ratio", "q_in"])
E = pd.DataFrame({"q0_ho": _p[("test_utility_ratio", 0)], "q768_ho": _p[("test_utility_ratio", 768)],
                  "q0_in": _p[("q_in", 0)], "q768_in": _p[("q_in", 768)]}).reset_index().rename(columns={"repeat_seed": "order"})
E["gain_ho"] = E.q768_ho - E.q0_ho; E["gain_in"] = E.q768_in - E.q0_in; E["source"] = "ebay"
def feats_ebay(i):
    s = S[i]; eu = s["eu"]
    return dict(K=s["K"], dig_v=digits(s["v"]), dig_bid=float(np.mean([digits(b) for b in s["bids"]])), p_decimals=4,
                eu_cv=float(np.std(eu) / np.mean(eu)) if np.mean(eu) > 0 else np.nan, gap12=s["gap12"])
F = pd.DataFrame([dict(scenario_id=i, **feats_ebay(i)) for i in S])
E = E.merge(F, on="scenario_id"); E["sid"] = "e" + E.scenario_id.astype(str)
# ---- synthetic decisions (exact payoffs)
sy = pd.read_csv("../synthetic_first_tol0.02_floor1e-4.csv")
_y = sy[sy.budget.isin([0, 768])].pivot_table(index=["model", "scenario_id", "order_seed"], columns="budget", values="utility_ratio")
Y = pd.DataFrame({"q0_in": _y[0], "q768_in": _y[768]}).reset_index().rename(columns={"order_seed": "order"}); Y["gain_in"] = Y.q768_in - Y.q0_in; Y["source"] = "synthetic"
meta = sy.drop_duplicates("scenario_id").set_index("scenario_id")
def feats_syn(i):
    r = meta.loc[i]; c = json.loads(r.displayed); bids = np.array([b for b, _ in c]); p = np.array([round(q, 4) for _, q in c])
    eu = p * (r.v - bids); srt = np.sort(eu)[::-1]
    return dict(K=int(r.K), dig_v=digits(r.v), dig_bid=float(np.mean([digits(b) for b in bids])), p_decimals=4,
                eu_cv=float(np.std(eu) / np.mean(eu)) if np.mean(eu) > 0 else np.nan, gap12=float((srt[0] - srt[1]) / srt[0]) if srt[0] > 0 else np.nan)
FS = pd.DataFrame([dict(scenario_id=i, **feats_syn(i)) for i in meta.index])
Y = Y.merge(FS, on="scenario_id"); Y["sid"] = "s" + Y.scenario_id.astype(str)
out = {"features": {}}
for name, df in [("ebay", F), ("synthetic", FS)]:
    out["features"][name] = {c: dict(median=float(df[c].median()), mean=float(df[c].mean())) for c in ["dig_v", "dig_bid", "eu_cv", "gap12", "K"]}
# ---- completion at 512 / 768 (parser v2)
tf = load_parse(0.02)
sy["K"] = sy.K.astype(int); sy["fp"] = sy.first_pos.map(json.loads); sy["complete"] = sy.fp.map(len) == sy.K
out["completion"] = {}
for m in MODELS:
    out["completion"][m] = {f"ebay_{b}": float(tf[(tf.model == m) & (tf.budget == b)].complete.mean()) for b in [512, 768]}
    out["completion"][m].update({f"synthetic_{b}": float(sy[(sy.model == m) & (sy.budget == b)].complete.mean()) for b in [512, 768]})
    # gap recovered, in-sample eBay vs synthetic
    e = E[E.model == m]; y = Y[Y.model == m]
    out["completion"][m]["gap_in_ebay"] = float((e.q768_in.mean() - e.q0_in.mean()) / (1 - e.q0_in.mean()))
    out["completion"][m]["gap_synthetic"] = float((y.q768_in.mean() - y.q0_in.mean()) / (1 - y.q0_in.mean()))
    out["completion"][m]["q0_ebay_in"] = float(e.q0_in.mean()); out["completion"][m]["q0_synthetic"] = float(y.q0_in.mean())
# ---- regressions: pooled in-sample gain
A = pd.concat([E[["model", "sid", "source", "gain_in", "K", "dig_v", "dig_bid", "eu_cv", "gap12"]],
               Y[["model", "sid", "source", "gain_in", "K", "dig_v", "dig_bid", "eu_cv", "gap12"]]]).dropna()
A["syn"] = (A.source == "synthetic").astype(int)
cl = pd.factorize(A.sid)[0]
m1 = smf.ols("gain_in ~ syn + C(model)", data=A).fit(cov_type="cluster", cov_kwds={"groups": cl})
m2 = smf.ols("gain_in ~ syn + C(model) + C(K) + dig_v + dig_bid + eu_cv + gap12", data=A).fit(cov_type="cluster", cov_kwds={"groups": cl})
m3 = smf.ols("gain_in ~ C(model) + C(K) + dig_v + dig_bid + eu_cv + gap12", data=A[A.syn == 0]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(A[A.syn == 0].sid)[0]})
pick = lambda mod: {k: dict(coef=float(mod.params[k]), se=float(mod.bse[k]), p=float(mod.pvalues[k])) for k in mod.params.index if not k.startswith("C(model)") and k != "Intercept"}
out["regression"] = dict(source_only=pick(m1), with_features=pick(m2), ebay_only=pick(m3), r2=[float(m1.rsquared), float(m2.rsquared), float(m3.rsquared)])
# also the 768 utility ratio of complete vs unfinished traces in synthetic, to see whether unfinished synthetic traces still choose well
out["synthetic_q_by_completion_768"] = {m: dict(complete=float(sy[(sy.model == m) & (sy.budget == 768) & sy.complete].utility_ratio.mean()),
                                                unfinished=float(sy[(sy.model == m) & (sy.budget == 768) & ~sy.complete].utility_ratio.mean()))
                                        for m in MODELS}
json.dump(out, open(f"{OUTDIR}/t4_synthetic_vs_ebay.json", "w"), indent=1, default=float)
print(json.dumps(out["features"], indent=0))
for m in MODELS: print(LAB[m], {k: round(v, 3) for k, v in out["completion"][m].items()}, {k: round(v, 3) for k, v in out["synthetic_q_by_completion_768"][m].items()})
for k, v in out["regression"].items():
    print(k, {kk: (round(vv["coef"], 4), round(vv["p"], 4)) for kk, vv in v.items()} if isinstance(v, dict) else v)
