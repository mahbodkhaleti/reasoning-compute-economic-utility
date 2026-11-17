"""A1 (cont.): verifiers that read the trace without doing any arithmetic (extract7.py output, ../trace_reader7.csv).
conclusion: the trace's last stated choice ('the highest is Option C', 'choose C', 'best bid is $250').
segment:    argmax of the last result number written in each option's segment, if >= K-1 options have one.
segment2:   the same rule once >= 2 options have one (reads partial comparisons).
combined:   conclusion if present, else segment, else the model's answer.
Validated against the annotators' recorded values on the 150 annotated traces."""
import json, numpy as np, pandas as pd
from common_rev import *
rng = np.random.default_rng(20261117)
d = records(); g = d[d.decoding == "greedy"]; q0 = g[g.budget == 0].groupby("model").test_utility_ratio.mean()
tf = load_parse(0.02)
rd = pd.read_csv("../trace_reader7.csv")
tf = tf.merge(rd, on=["model", "scenario_id", "repeat_seed", "budget"], how="left")
tf["seg"] = tf.seg_values.fillna("{}").map(lambda s: {float(k): v for k, v in json.loads(s).items()})
def pick_seg(r):
    if len(r.seg) >= r.K - 1: return max(r.seg.items(), key=lambda kv: kv[1])[0]
    return np.nan
tf["seg_bid"] = [pick_seg(r) for r in tf.itertuples()]
tf["v_conc"] = np.where(tf.conclusion_bid.notna(), tf.conclusion_bid, tf.chosen_bid)
tf["v_seg"] = np.where(tf.seg_bid.notna(), tf.seg_bid, tf.chosen_bid)
tf["v_seg2"] = [max(r.seg.items(), key=lambda kv: kv[1])[0] if len(r.seg) >= 2 else r.chosen_bid for r in tf.itertuples()]
tf["v_comb"] = np.where(tf.conclusion_bid.notna(), tf.conclusion_bid, tf.v_seg)
for c in ("v_conc", "v_seg", "v_seg2", "v_comb"): tf["q_" + c] = [q_of(i, b) for i, b in zip(tf.scenario_id, tf[c])]
def gr(z, c, m): return (z[c].mean() - q0[m]) / (z.qcalc.mean() - q0[m])
R = {"by_model": {}}
for m in MODELS:
    rows = []
    for bud in [0, 64, 128, 256, 512, 768, -1]:
        z = tf[(tf.model == m) & (tf.budget == bud)]
        rows.append(dict(budget=bud, has_conclusion=float(z.conclusion_bid.notna().mean()), seg_cov=float((z.seg.map(len) / z.K).mean()),
                         conc_agrees_answer=float(np.isclose(z.conclusion_bid, z.chosen_bid)[z.conclusion_bid.notna()].mean()) if z.conclusion_bid.notna().any() else None,
                         **{f"q_{c}": float(z["q_" + c].mean()) for c in ("read", "v_conc", "v_seg", "v_seg2", "v_comb", "be")},
                         **{f"gap_{c}": float(gr(z, "q_" + c, m)) for c in ("read", "v_conc", "v_seg", "v_seg2", "v_comb", "be")}))
    z = tf[(tf.model == m) & (tf.budget == 768)].reset_index(drop=True)
    bs = np.array([[gr(z.iloc[ids], "q_" + c, m) for c in ("v_conc", "v_seg", "v_seg2", "v_comb")] for ids in (family_boot_ids(z.fam.values, rng) for _ in range(2000))])
    R["by_model"][m] = dict(by_budget=rows, ci768={c: [float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))] for j, c in enumerate(("v_conc", "v_seg", "v_seg2", "v_comb"))},
                            min_diff={c: float(min(r[f"q_{c}"] - r["q_read"] for r in rows)) for c in ("v_conc", "v_seg", "v_seg2", "v_comb")},
                            diff_by_budget={c: {r["budget"]: float(r[f"q_{c}"] - r["q_read"]) for r in rows} for c in ("v_conc", "v_seg", "v_seg2", "v_comb")})
# validation of segment values against annotators
vs = json.load(open("../validation_sample.json")); L = {}
for p in (0, 1):
    for x in json.load(open(f"/tmp/www2027/judge/label/labels_{p}.json")): L[x["item"]] = x
key = tf.set_index(["model", "scenario_id", "repeat_seed", "budget"])
n_opt = n_hit = n_found = n_lab = 0
for k, r in enumerate(vs):
    kk = (r["model"], int(r["scenario_id"]), int(r["repeat_seed"]), int(r["budget"]))
    if kk not in key.index: continue
    seg = key.loc[kk, "seg"]; seg = seg.iloc[0] if isinstance(seg, pd.Series) else seg
    lab = {o["label"]: o for o in L[k]["options"]}
    for j, b in enumerate(r["option_order"]):
        o = lab["ABCDEFG"[j]]
        if o["stated"] and o["value"] is not None:
            n_lab += 1
            if float(b) in seg:
                n_found += 1; n_hit += int(abs(seg[float(b)] - o["value"]) <= 0.02 * abs(o["value"]) + 0.05)
R["validation"] = dict(annotated_stated_options=n_lab, found=n_found, recall=n_found / max(n_lab, 1), value_match_when_found=n_hit / max(n_found, 1))
json.dump(R, open(f"{OUTDIR}/t9b_trace_readers.json", "w"), indent=1, default=float)
print(R["validation"])
for m, v in R["by_model"].items():
    b = {r["budget"]: r for r in v["by_budget"]}
    print(LAB[m], "conc", round(b[768]["has_conclusion"], 2), "agree", round(b[768]["conc_agrees_answer"], 2), "segcov", round(b[768]["seg_cov"], 2),
          "gap768 read/conc/seg/comb/oracle", [round(b[768][f"gap_{c}"], 2) for c in ("read", "v_conc", "v_seg", "v_seg2", "v_comb", "be")],
          {c: [round(x, 2) for x in ci] for c, ci in v["ci768"].items()}, "mindiff", {c: round(x, 3) for c, x in v["min_diff"].items()},
          "uncon", [round(b[-1][f"gap_{c}"], 2) for c in ("read", "v_comb")], "512", [round(b[512][f"gap_{c}"], 2) for c in ("read", "v_comb")])
