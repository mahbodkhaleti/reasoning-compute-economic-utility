"""Item 5: what predicts passing over the best option? One logistic regression across all six models.
Sample: 768-token traces in which the trace has written the best option's value and ranks it first (so the answer
could have read it off): categories Incompl., Passed and right. Outcome: the answer is another bid.
Predictors: relative profit gap between best and runner-up (difficulty), normalised list position of the best option
(0 first, 1 last), K = 7, log private value relative to the product median, product, best option is also the
highest bid on the menu, list finished, and model. Interactions model x position and model x best-is-highest let the
position and highest-bid effects differ by model.
Main fit: logit with standard errors clustered by scenario. Check: Bayesian mixed logit with a scenario random
intercept (statsmodels BinomialBayesMixedGLM, variational fit)."""
import sys; sys.path.insert(0, "/tmp/www2027/analysis/rev25")
from r_common import *
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
z = tf[tf.budget == 768].copy()
z = z[z.cls.isin(["incomplete", "passed", "right"])]
z = z[[(c in hv) and not any(hv[o][0] >= hv[c][0] for o in hv if o != c) for hv, c in zip(z.hv, z.calc)]].copy()
med = {p: np.median([S[i]["v"] for i in S if S[i]["product"] == p]) for p in PRODUCTS}
z["passed_over"] = (~np.isclose(z.chosen_bid, z.calc)).astype(int)
z["gap12"] = z.scenario_id.map(lambda i: S[i]["gap12"])
z["pos"] = [o.index(c) / (len(o) - 1) for o, c in zip(z.order_list, z.calc)]
z["K7"] = (z.K == 7).astype(int)
z["logv"] = [np.log(S[i]["v"] / med[S[i]["product"]]) for i in z.scenario_id]
z["product"] = z.scenario_id.map(lambda i: S[i]["product"].split()[0])
z["best_is_high"] = [int(np.isclose(S[i]["calc"], S[i]["bids"].max())) for i in z.scenario_id]
z["finished"] = [int(len(hv) >= K) for hv, K in zip(z.hv, z.K)]
z["mod"] = z.model.map(LAB)
z["sid"] = pd.factorize(z.scenario_id)[0]
form = "passed_over ~ gap12 + pos + K7 + logv + C(product) + best_is_high + finished + C(mod, Treatment('Qwen3-8B'))"
fit = smf.logit(form, data=z).fit(disp=0, cov_type="cluster", cov_kwds={"groups": z.sid})
R = {"n": int(len(z)), "events": int(z.passed_over.sum()), "pooled": {}}
for k in fit.params.index:
    lo, hi = fit.conf_int().loc[k]
    R["pooled"][k] = dict(OR=float(np.exp(fit.params[k])), ci=[float(np.exp(lo)), float(np.exp(hi))], p=float(fit.pvalues[k]))
# per-model slopes for position and best-is-highest (interaction model)
form2 = "passed_over ~ gap12 + K7 + logv + C(product) + finished + C(mod) + C(mod):pos + C(mod):best_is_high"
fit2 = smf.logit(form2, data=z).fit(disp=0, cov_type="cluster", cov_kwds={"groups": z.sid}, maxiter=200)
R["by_model"] = {}
for k in fit2.params.index:
    if ":pos" in k or ":best_is_high" in k:
        lo, hi = fit2.conf_int().loc[k]
        R["by_model"][k] = dict(OR=float(np.exp(fit2.params[k])), ci=[float(np.exp(lo)), float(np.exp(hi))], p=float(fit2.pvalues[k]))
# difficulty-only vs with best_is_high: does the highest-bid pull survive controlling for difficulty?
R["share_best_is_high"] = float(z.best_is_high.mean())
# mixed-effects check
try:
    mm = BinomialBayesMixedGLM.from_formula(form, {"scen": "0 + C(sid)"}, data=z).fit_vb()
    R["mixed_vb"] = {k: dict(OR=float(np.exp(v)), sd=float(s)) for k, v, s in zip(mm.model.exog_names, mm.fe_mean, mm.fe_sd)}
    R["mixed_vb_scenario_sd"] = float(np.exp(mm.vcp_mean[0]))
except Exception as e:
    R["mixed_vb_error"] = str(e)
dump25("r5_passover_regression", R)
print("n", R["n"], "events", R["events"])
for k, v in R["pooled"].items(): print(f"  {k:55s} OR {v['OR']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}] p={v['p']:.3g}")
for k, v in R["by_model"].items(): print(f"  {k:55s} OR {v['OR']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}] p={v['p']:.3g}")
if "mixed_vb" in R:
    for k in ["gap12", "pos", "K7", "logv", "best_is_high", "finished"]: print("  mixed", k, round(R["mixed_vb"][k]["OR"], 2), "sd(log)", round(R["mixed_vb"][k]["sd"], 2))
    print("  scenario RE sd", round(R["mixed_vb_scenario_sd"], 2))
