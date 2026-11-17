"""Routing evaluated on the utility-ratio gain (not dollars).
Same folds and signals as batch3. Two variants:
  (a) dollar-tuned: thresholds picked on construction-split DOLLAR gain (as in Table 2), scored on held-out RATIO gain kept
  (b) ratio-tuned: thresholds picked on construction-split RATIO gain, scored on held-out ratio gain kept
Also ranking areas on the held-out ratio gain."""
import json, numpy as np, pandas as pd
rng = np.random.default_rng(20261107)
exec(open("batch3_reviewer_checks.py").read().split("# ======================= A. routing signals")[0])
W = g[g.budget.isin([0, 768])].pivot_table(index=["model", "scenario_id", "repeat_seed"], columns="budget",
        values=["test_utility", "chosen_bid", "constrained_option_probability", "test_utility_ratio"]).reset_index()
W.columns = ["model", "scenario_id", "order", "b0", "b768", "p0", "p768", "te0", "te768", "q0", "q768"]
def tr(sid, b):
    s = S[sid]; k = np.where(np.isclose(s["bids"], b))[0][0]; return s["train_p"][k] * (s["v"] - b)
W["tr0"] = [tr(i, b) for i, b in zip(W.scenario_id, W.b0)]; W["tr768"] = [tr(i, b) for i, b in zip(W.scenario_id, W.b768)]
W["trstar"] = W.scenario_id.map(lambda i: float(np.max(S[i]["train_p"] * (S[i]["v"] - S[i]["bids"]))))
W["trq0"] = W.tr0 / W.trstar; W["trq768"] = W.tr768 / W.trstar
W["family"] = W.scenario_id.map(lambda i: S[i]["family"]); W["product"] = W.scenario_id.map(lambda i: S[i]["product"])
W["value"] = W.scenario_id.map(lambda i: S[i]["v"]); W["stakes"] = W.trstar; W["neg_conf"] = -W.p0
prod_scale = {p: float(np.median(price[(p, "train")])) for p in PRODUCTS}; W["product_scale"] = W["product"].map(prod_scale)
fold_of = {}
for p in PRODUCTS:
    fl = [f for f in FAM if PROD_OF[f] == p]; perm = rng.permutation(len(fl))
    for i, k in enumerate(perm): fold_of[fl[k]] = i % 6
W["fold"] = W.family.map(fold_of)
def pick(score, a0, a1, target=0.9):
    gain = (a1 - a0).sum(); order = np.argsort(-score, kind="stable"); cum = np.cumsum((a1 - a0)[order])
    ok = np.where(cum >= target * gain)[0]; k = ok[0] if len(ok) else len(score) - 1
    return score[order][k]
def area(score, a0, a1):
    order = np.argsort(-score, kind="stable"); cum = np.r_[0, np.cumsum((a1 - a0)[order])]; n = len(score)
    grid = np.linspace(0, 1, 101); kept = np.array([cum[int(round(q * n))] / cum[-1] for q in grid])
    return float(np.trapezoid(kept, grid)), float(kept[50])
out = {}
for m in MODELS:
    x = W[W.model == m].reset_index(drop=True); jit = rng.uniform(0, 1e-7, len(x)); res = {}
    for name, col in {"value": "value", "product": "product_scale", "confidence": "neg_conf", "stakes": "stakes"}.items():
        sc_ = x[col].values.astype(float) + jit; r = {}
        for tune, (a0, a1) in {"dollar_tuned": ("tr0", "tr768"), "ratio_tuned": ("trq0", "trq768")}.items():
            esc = np.zeros(len(x), bool)
            for k in range(6):
                trm = (x.fold != k).values
                t = pick(sc_[trm], x[a0].values[trm], x[a1].values[trm]); esc[~trm] = sc_[~trm] >= t
            kq = (np.where(esc, x.q768, x.q0).mean() - x.q0.mean()) / (x.q768.mean() - x.q0.mean())
            kd = (np.where(esc, x.te768, x.te0).mean() - x.te0.mean()) / (x.te768.mean() - x.te0.mean())
            bs = []
            for _ in range(1000):
                ids = np.concatenate([np.where(x.family.values == f)[0] for p in PRODUCTS for f in rng.choice([f for f in FAM if PROD_OF[f] == p], 18, replace=True)])
                bs.append((np.where(esc[ids], x.q768.values[ids], x.q0.values[ids]).mean() - x.q0.values[ids].mean()) / (x.q768.values[ids].mean() - x.q0.values[ids].mean()))
            r[tune] = dict(saved=float(1 - esc.mean()), kept_ratio=float(kq), kept_ratio_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], kept_dollar=float(kd))
        r["area_ratio"], r["kept50_ratio"] = area(sc_, x.q0.values, x.q768.values)
        res[name] = r
    out[m] = res
json.dump(out, open("out/routing_ratio.json", "w"), indent=1)
for m in MODELS:
    print(LAB[m])
    for k, v in out[m].items():
        print("  %-10s dollar-tuned saved %.2f keptQ %.2f [%.2f,%.2f] | ratio-tuned saved %.2f keptQ %.2f [%.2f,%.2f] | areaQ %.2f k50 %.2f" % (
            k, v["dollar_tuned"]["saved"], v["dollar_tuned"]["kept_ratio"], *v["dollar_tuned"]["kept_ratio_ci"],
            v["ratio_tuned"]["saved"], v["ratio_tuned"]["kept_ratio"], *v["ratio_tuned"]["kept_ratio_ci"], v["area_ratio"], v["kept50_ratio"]))
