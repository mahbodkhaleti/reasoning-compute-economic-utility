"""F1 with filtered trace values: a product match counts only if its probability equals the
displayed probability of that bid (within 2% relative); label summaries ('B is $20.64') always count.
Value per option = last kept match. Trace-implied bid = argmax over options when every option has a value."""
import json, numpy as np, pandas as pd
exec(open("batch4_judge_r1.py").read().split("# ================= F1")[0])
t = pd.read_csv("trace_matches.csv")
t["m"] = t["values"].map(json.loads); t["K"] = t.scenario_id.map(lambda i: S[i]["K"])
def values(row):
    s = S[row.scenario_id]; dp = dict(zip(s["bids"], s["dp"])); out = {}
    for b, p, r, pos, kind in sorted(row.m, key=lambda z: z[3]):
        if kind == "prod" and (p is None or abs(p - dp[float(b)]) > 0.02 * dp[float(b)] + 1e-9): continue
        out[float(b)] = r
    return out
t["vals"] = t.apply(values, axis=1)
t["complete"] = [len(v) == k for v, k in zip(t.vals, t.K)]
t["implied"] = [max(v.items(), key=lambda kv: kv[1])[0] if c else np.nan for v, c in zip(t.vals, t.complete)]
t["calc"] = t.scenario_id.map(lambda i: S[i]["calc"])
t["q_read"] = [q_of(i, b) for i, b in zip(t.scenario_id, t.chosen_bid)]
t["q_impl"] = [q_of(i, b) if not np.isnan(b) else np.nan for i, b in zip(t.scenario_id, t.implied)]
t["qcalc"] = t.scenario_id.map(lambda i: q_of(i, S[i]["calc"]))
out = {}
for m in MODELS:
    x = t[t.model == m]; res = {}
    for bud in [256, 512, 768, -1]:
        y = x[x.budget == bud]; c = y[y.complete]
        res[bud] = dict(complete=float(y.complete.mean()), q_read=float(y.q_read.mean()),
                        q_read_complete=float(c.q_read.mean()) if len(c) else None, q_impl_complete=float(c.q_impl.mean()) if len(c) else None,
                        impl_is_calc=float(np.isclose(c.implied, c.calc).mean()) if len(c) else None,
                        read_eq_impl=float(np.isclose(c.implied, c.chosen_bid).mean()) if len(c) else None,
                        q_read_incomplete=float(y[~y.complete].q_read.mean()) if (~y.complete).any() else None)
    y = x[x.budget == 768]
    short = (y.qcalc - y.q_read); tot = short.sum()
    mis = ~np.isclose(y.chosen_bid, y.calc)
    inc = mis & ~y.complete
    intr = mis & y.complete & ~np.isclose(y.implied, y.calc)
    rd = mis & y.complete & np.isclose(y.implied, y.calc)
    res["decomp"] = dict(loss_incomplete=float(short[inc].sum() / tot), loss_in_trace=float(short[intr].sum() / tot), loss_readout=float(short[rd].sum() / tot),
                         n_mis=int(mis.sum()))
    out[m] = res
json.dump(out, open("out/f1_filtered.json", "w"), indent=1, default=float)
for m in MODELS:
    print(LAB[m], {b: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[m][b].items()} for b in [512, 768, -1]})
    print("   ", {k: round(v, 3) if isinstance(v, float) else v for k, v in out[m]["decomp"].items()})
