"""Robustness of the Table 2 split to the parser.
G1  Rebuild the hybrid written values and the four-way split at match tolerances 1%, 2% (paper) and 5%.
G2  Worst-case check for Passed: count a trace as Passed only if every option's value is found by BOTH the exact
    matcher (2%) and the text reader; move every other Passed trace to Incomplete (the least favourable reading
    for the claim that complete traces are answered against). Report the split under that rule.
G3  Share of Table 2 options whose written value comes from the exact matcher vs the text reader alone."""
import json, numpy as np, pandas as pd
_src = open("t10b_hybrid_split.py").read()
exec(_src[:_src.index('R = {"forced768"')])
base = tf[["model", "scenario_id", "repeat_seed", "budget", "vals", "chosen_bid", "fam"]].copy()
def classify(hv, i, K, chosen, strict_ok=True):
    s = S[i]; calc = s["calc"]; eu = lambda o: s["eu"][np.isclose(s["bids"], o)][0]
    if np.isclose(chosen, calc): return "right"
    if calc not in hv: return "not_reached"
    if any(hv[o][0] >= hv[calc][0] for o in hv if o != calc and eu(o) < eu(calc)): return "arith"
    if len(hv) < K: return "incomplete"
    return "passed" if strict_ok else "incomplete"
R = {"G1": {}, "G2": {}, "G3": {}}
for tol in (0.01, 0.02, 0.05):
    tp = load_parse(tol)[["model", "scenario_id", "repeat_seed", "budget", "fp", "K", "q_read", "qcalc"]]
    z = base.merge(tp, on=["model", "scenario_id", "repeat_seed", "budget"])
    z["hv"] = [hybrid(vv, fp, i) for vv, fp, i in zip(z.vals, z.fp, z.scenario_id)]
    z["gap"] = z.qcalc - z.q_read
    z["cls"] = [classify(hv, i, K, b) for hv, i, K, b in zip(z.hv, z.scenario_id, z.K, z.chosen_bid)]
    for m in MODELS:
        w = z[(z.model == m) & (z.budget == 768)]; T = w.gap.sum()
        R["G1"].setdefault(m, {})[str(tol)] = {k: float(w.gap[w.cls == k].sum() / T) for k in cats}
    if tol == 0.02:
        both = [all((b in vv) and any(np.isclose(float(x), b) for x in fp) for b in S[i]["bids"]) for vv, fp, i in zip(z.vals, z.fp, z.scenario_id)]
        z["cls_strict"] = [classify(hv, i, K, b, ok) for hv, i, K, b, ok in zip(z.hv, z.scenario_id, z.K, z.chosen_bid, both)]
        for m in MODELS:
            w = z[(z.model == m) & (z.budget == 768)]; T = w.gap.sum()
            R["G2"][m] = {k: float(w.gap[w.cls_strict == k].sum() / T) for k in cats}
            src = [k for hv in w.hv for (_, k) in hv.values()]
            R["G3"][m] = dict(n_values=len(src), share_exact=float(np.mean([k == "true" for k in src])))
json.dump(R, open(f"{OUTDIR}/t16_parser_robustness.json", "w"), indent=1)
f = lambda x: round(100 * x)
for m in MODELS:
    print(LAB[m], {t: [f(v[k]) for k in cats] for t, v in R["G1"][m].items()}, "strict", [f(R["G2"][m][k]) for k in cats], "exact", round(R["G3"][m]["share_exact"], 3))
print("order:", cats)
