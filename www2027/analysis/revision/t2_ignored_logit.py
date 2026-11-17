"""Task 2: when does a complete trace's answer ignore its own best evaluated option?
Sample: greedy traces at 768 tokens and the unconstrained run that evaluate every option (parser, 2%).
Outcome: answer != best evaluated option.
Predictors: displayed position of the best option (1 = first), relative top-two margin of displayed expected profit,
K, product, option-order seed, budget, and whether the best option is also the highest-probability bid.
(The user's spec listed 'chosen option is the highest-probability bid'; that is part of the outcome, so it is
replaced by the best option's own highest-probability status, and the share of ignored answers that are the
highest-probability bid is reported descriptively.)
Logit by model family (Qwen pooled with model fixed effects), SEs clustered by scenario.
"""
import json, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from common_rev import *

tf = load_parse(0.02)
tm = pd.read_csv("../trace_matches.csv", usecols=["model", "scenario_id", "repeat_seed", "budget", "option_order"])
tf = tf.merge(tm, on=["model", "scenario_id", "repeat_seed", "budget"], how="left")
tf["order_list"] = tf.option_order.map(json.loads)
y = tf[tf.budget.isin([768, -1]) & tf.complete].copy()
y["ignored"] = (~np.isclose(y.chosen_bid, y.best_eval_bid)).astype(int)
y["best_pos"] = [lst.index(b) + 1 for lst, b in zip(y.order_list, y.best_eval_bid)]
y["best_pos_rel"] = (y.best_pos - 1) / (y.K - 1)
y["gap12"] = y.scenario_id.map(lambda i: S[i]["gap12"])
y["best_is_hp"] = [int(np.isclose(b, S[i]["hp"])) for i, b in zip(y.scenario_id, y.best_eval_bid)]
y["chosen_is_hp"] = [int(np.isclose(b, S[i]["hp"])) for i, b in zip(y.scenario_id, y.chosen_bid)]
y["product"] = y.scenario_id.map(lambda i: S[i]["product"].split()[0])
y["order"] = y.repeat_seed.astype(str); y["bud"] = y.budget.astype(str)
FAMILY = {"DeepSeek": ["deepseek_r1_llama_8b"], "Phi": ["phi4_mini_reasoning"], "Qwen": ["qwen3_4b", "qwen3_8b", "qwen3_14b"],
          "Granite": ["granite_3_2_8b_instruct"]}
out = {}
for fam, ms in FAMILY.items():
    z = y[y.model.isin(ms)].reset_index(drop=True)
    f = "ignored ~ best_pos_rel + gap12 + C(K) + C(product) + C(order) + C(bud) + best_is_hp" + (" + C(model)" if len(ms) > 1 else "")
    mod = smf.glm(f, data=z, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(z.scenario_id)[0]})
    coefs = {k: dict(coef=float(mod.params[k]), se=float(mod.bse[k]), p=float(mod.pvalues[k]), odds_ratio=float(np.exp(mod.params[k])))
             for k in mod.params.index if not k.startswith("C(model)") and k != "Intercept"}
    ign = z[z.ignored == 1]
    out[fam] = dict(n=int(len(z)), ignored_rate=float(z.ignored.mean()),
                    ignored_rate_by_best_pos_first_vs_last=[float(z[z.best_pos == 1].ignored.mean()), float(z[z.best_pos == z.K].ignored.mean())],
                    ignored_rate_by_best_is_hp=[float(z[z.best_is_hp == 0].ignored.mean()), float(z[z.best_is_hp == 1].ignored.mean())],
                    share_ignored_choosing_hp=float(ign.chosen_is_hp.mean()) if len(ign) else None,
                    share_ignored_choosing_highest_bid=float(np.mean([np.isclose(b, S[i]["bids"].max()) for i, b in zip(ign.scenario_id, ign.chosen_bid)])) if len(ign) else None,
                    coefs=coefs)
json.dump(out, open(f"{OUTDIR}/t2_ignored_logit.json", "w"), indent=1, default=float)
for fam, r in out.items():
    print(fam, "n", r["n"], "ignored", round(r["ignored_rate"], 3), "first/last", [round(a, 3) for a in r["ignored_rate_by_best_pos_first_vs_last"]],
          "best_is_hp no/yes", [round(a, 3) for a in r["ignored_rate_by_best_is_hp"]], "ign->hp", round(r["share_ignored_choosing_hp"], 3))
    for k, c in r["coefs"].items(): print("   %-32s OR %.2f  p %.4f" % (k, c["odds_ratio"], c["p"]))
