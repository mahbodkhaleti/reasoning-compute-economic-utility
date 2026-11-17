"""Round-7 judge fixes (CPU, saved data only). Inputs: ../segments_hist.csv.gz and ../segments_syn.csv.gz (extract8.py).
B1  Text reader with no arithmetic, windows 300/600/1000 chars, thresholds >=2 / >=K-1; eBay traces and, as data never
    used to design the reader, the synthetic traces.
B2  Decomposition on stated values only (no true-value matching): Not reached (best never written), Arith. (the trace's
    own numbers do not put the best option first, including a miscomputed best), Incomplete (best written and ranked
    first, not all options written, answer differs), Passed (all written, best ranked first, answer differs).
    Assigned in that order. Forced 768, unforced traces that stopped on their own, and synthetic 768.
B3  Interruption test: in forced-768 traces where the best option is written and ranked first but not all options are
    written, does the answer pick it more often the more tokens remain after its value was written?
B4  Per-option arithmetic error: annotator-recorded stated values vs true expected profit (150 annotated traces), and
    reader-read values vs true.
B5  Difficulty: passed-over rate in complete traces vs best-runner-up gap, K, product, contiguous vs scattered menu;
    near-ties among passed-over cases; chance rate 1/(K-1) for repeating the budget-0 answer.
B6  Monitoring: when the reader disagrees with the answer, how often is the answer wrong?
B7  Economics of wrong answers at 768: over- vs underbidding, expected payment to the seller, win probability, and
    expected profit lost as a share of the private value, by product.
B8  Intervals for all Table 1 columns.
"""
import json, numpy as np, pandas as pd, statsmodels.formula.api as smf
from common_rev import *
rng = np.random.default_rng(20261118)
d = records(); g = d[d.decoding == "greedy"]; q0 = g[g.budget == 0].groupby("model").test_utility_ratio.mean()
tf = load_parse(0.02)
seg = pd.read_csv("../segments_hist.csv.gz").rename(columns={"seed": "repeat_seed", "n_chars": "n_chars_seg"})
tf = tf.merge(seg, on=["model", "scenario_id", "repeat_seed", "budget"], how="left")
tf["segs"] = tf.segments.fillna("[]").map(json.loads)

def read_vals(segs, v, info, W):
    """info: {bid: p}. Last result number in each option segment, skipping the inputs (bid, v, v-b, p)."""
    out = {}
    for b, st, nxt, nums in segs:
        b = float(b)
        if b not in info: continue
        skip = {round(b, 4), round(v, 4), round(v - b, 4)}
        last = None
        for x, pos in nums:
            if pos >= nxt or pos >= st + W: break
            if round(x, 4) in skip or abs(x - info[b]) < 5e-5: continue
            last = (x, pos)
        if last is not None: out[b] = last
    return out
def pick(vals, need, fallback):
    return max(vals.items(), key=lambda kv: kv[1][0])[0] if len(vals) >= need and len(vals) > 0 else fallback
R = {k: {} for k in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]}
info_h = {i: dict(zip(S[i]["bids"], S[i]["dp"])) for i in S}
for W in (300, 600, 1000):
    tf[f"vals{W}"] = [read_vals(sg, S[i]["v"], info_h[i], W) for sg, i in zip(tf.segs, tf.scenario_id)]
tf["vals"] = tf["vals600"]
def gr(z, c, m): return (z[c].mean() - q0[m]) / (z.qcalc.mean() - q0[m])
for W in (300, 600, 1000):
    for nm, needf in (("ge2", lambda K: 2), ("km1", lambda K: K - 1)):
        col = f"q_{nm}_{W}"
        tf[col] = [q_of(i, pick(vv, needf(K), b)) for i, vv, K, b in zip(tf.scenario_id, tf[f"vals{W}"], tf.K, tf.chosen_bid)]
for m in MODELS:
    res = {}
    for bud in (512, 768, -1):
        z = tf[(tf.model == m) & (tf.budget == bud)]
        res[str(bud)] = {c[2:]: float(gr(z, c, m)) for c in tf.columns if c.startswith("q_ge2_") or c.startswith("q_km1_")}
        res[str(bud)]["answer"] = float(gr(z, "q_read", m))
    res["min_diff_ge2_600"] = float(min((tf[(tf.model == m) & (tf.budget == b_)]["q_ge2_600"] - tf[(tf.model == m) & (tf.budget == b_)].q_read).mean()
                                     for b_ in (0, 64, 128, 256, 512, 768, -1)))
    R["B1"][m] = res

# synthetic
sy = pd.read_csv("../synthetic_first_tol0.02_floor1e-4.csv")
ss = pd.read_csv("../segments_syn.csv.gz").rename(columns={"seed": "order_seed"})
sy = sy.merge(ss[["model", "scenario_id", "order_seed", "budget", "segments"]], on=["model", "scenario_id", "order_seed", "budget"], how="left")
sy["disp"] = sy.displayed.map(json.loads)
sy["pinfo"] = sy.disp.map(lambda L: {float(b): round(p, 4) for b, p in L})
sy["eu"] = [{float(b): p * (v - b) for b, p in L} for L, v in zip(sy.disp, sy.v)]
sy["calc"] = sy.eu.map(lambda e: max(e.items(), key=lambda kv: kv[1])[0])
sy["umax"] = sy.eu.map(lambda e: max(e.values()))
sy["vals"] = [read_vals(json.loads(sg) if isinstance(sg, str) else [], v, inf, 600) for sg, v, inf in zip(sy.segments, sy.v, sy.pinfo)]
def sq(e, b, um): return e[float(b)] / um
sy["q_read"] = [sq(e, b, u) for e, b, u in zip(sy.eu, sy.selected_bid, sy.umax)]
for nm, needf in (("ge2", lambda K: 2), ("km1", lambda K: K - 1)):
    sy[f"q_{nm}"] = [sq(e, pick(vv, needf(K), b), u) for e, vv, K, b, u in zip(sy.eu, sy.vals, sy.K, sy.selected_bid, sy.umax)]
sq0 = sy[sy.budget == 0].groupby("model").q_read.mean()
R["B1"]["synthetic"] = {}
for m in MODELS:
    rr = {}
    for bud in (512, 768):
        z = sy[(sy.model == m) & (sy.budget == bud)]
        rr[str(bud)] = {c: float((z[f"q_{c}"].mean() - sq0[m]) / (1 - sq0[m])) for c in ("read", "ge2", "km1")}
    rr["min_diff_ge2"] = float(min((sy[(sy.model == m) & (sy.budget == b_)].q_ge2 - sy[(sy.model == m) & (sy.budget == b_)].q_read).mean() for b_ in sorted(sy.budget.unique())))
    R["B1"]["synthetic"][m] = rr

# ---------------- B2 decomposition on stated values
def classify(vals, calc, eu_of, K, chosen):
    if np.isclose(chosen, calc): return "right"
    if calc not in vals: return "not_reached"
    if any(vals[o][0] >= vals[calc][0] for o in vals if o != calc and eu_of(o) < eu_of(calc)): return "arith"
    if len(vals) < K: return "incomplete"
    return "passed"
def decompose(z, gapcol, cls, fams):
    gap = z[gapcol].values; c = z[cls].values
    cats = ["not_reached", "arith", "incomplete", "passed"]
    def sh(ids):
        T = gap[ids].sum(); return [gap[ids][c[ids] == k].sum() / T for k in cats]
    pt = sh(np.arange(len(z))); bs = np.array([sh(family_boot_ids(fams, rng)) for _ in range(1000)])
    return {k: dict(share=float(pt[j]), ci=[float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))], n=int((c == k).sum())) for j, k in enumerate(cats)}
tf["cls"] = [classify(vv, S[i]["calc"], lambda o, i=i: S[i]["eu"][np.isclose(S[i]["bids"], o)][0], K, b) for vv, i, K, b in zip(tf.vals, tf.scenario_id, tf.K, tf.chosen_bid)]
tf["gap"] = tf.qcalc - tf.q_read
R["B2"]["forced768"] = {}; R["B2"]["natural_stop"] = {}; R["B2"]["synthetic768"] = {}
for m in MODELS:
    z = tf[(tf.model == m) & (tf.budget == 768)].reset_index(drop=True)
    R["B2"]["forced768"][m] = decompose(z, "gap", "cls", z.fam.values)
    zn = tf[(tf.model == m) & (tf.budget == -1) & (tf.reasoning_tokens < 1024)].reset_index(drop=True)
    R["B2"]["natural_stop"][m] = dict(n=int(len(zn)), q=float(zn.q_read.mean()), qcalc=float(zn.qcalc.mean()),
                                      wrong=float((zn.cls != "right").mean()),
                                      **({"split": decompose(zn, "gap", "cls", zn.fam.values)} if (zn.cls != "right").sum() >= 20 else {}))
sy["cls"] = [classify(vv, c, lambda o, e=e: e[o], K, b) for vv, c, e, K, b in zip(sy.vals, sy.calc, sy.eu, sy.K, sy.selected_bid)]
sy["gap"] = 1 - sy.q_read
sy["fam"] = sy.scenario_id
for m in MODELS:
    z = sy[(sy.model == m) & (sy.budget == 768)].reset_index(drop=True)
    gap = z.gap.values; c = z.cls.values; T = gap.sum()
    R["B2"]["synthetic768"][m] = {k: float(gap[c == k].sum() / T) for k in ["not_reached", "arith", "incomplete", "passed"]}

# ---------------- B3 interruption test
rows = []
for m in MODELS:
    z = tf[(tf.model == m) & (tf.budget == 768)]
    for r in z.itertuples():
        vv = r.vals; c = r.calc
        if c not in vv or len(vv) >= r.K: continue
        if any(vv[o][0] >= vv[c][0] for o in vv if o != c): continue
        pos = vv[c][1]; n = max(r.n_chars_seg, 1); after = n - pos
        rows.append(dict(model=m, tok_after=after / n * 768, right=int(np.isclose(r.chosen_bid, c)), K=r.K, sid=r.scenario_id))
B3 = pd.DataFrame(rows); B3["bin"] = pd.cut(B3.tok_after, [-1, 64, 192, 768], labels=["<64", "64-192", ">192"])
R["B3"]["by_bin"] = {m: {str(k): dict(n=int(len(x)), right=float(x.right.mean())) for k, x in B3[B3.model == m].groupby("bin", observed=True)} for m in MODELS}
R["B3"]["pooled"] = {str(k): dict(n=int(len(x)), right=float(x.right.mean())) for k, x in B3.groupby("bin", observed=True)}
lg = smf.logit("right ~ np.log1p(tok_after) + C(model) + C(K)", data=B3).fit(disp=0)
R["B3"]["logit_log_tokens_after"] = dict(coef=float(lg.params["np.log1p(tok_after)"]), p=float(lg.pvalues["np.log1p(tok_after)"]))
# compare: complete forced traces (not interrupted mid-list) with best ranked first
zc = tf[(tf.budget == 768) & (tf.cls.isin(["right", "passed"])) & (tf.vals.map(len) >= tf.K)]
R["B3"]["complete_written_right"] = {m: float((zc[zc.model == m].cls == "right").mean()) for m in MODELS}
R["B3"]["incomplete_best_first_right"] = {m: float(B3[B3.model == m].right.mean()) for m in MODELS}

# ---------------- B4 arithmetic error per option
vs = json.load(open("../validation_sample.json")); L = {}
for p in (0, 1):
    for x in json.load(open(f"/tmp/www2027/judge/label/labels_{p}.json")): L[x["item"]] = x
n = e = 0
for k, r in enumerate(vs):
    s = S[int(r["scenario_id"])]; lab = {o["label"]: o for o in L[k]["options"]}
    for j, b in enumerate(r["option_order"]):
        o = lab["ABCDEFG"[j]]
        if o["stated"] and o["value"] is not None:
            tv = s["eu"][np.isclose(s["bids"], float(b))][0]; n += 1; e += int(abs(o["value"] - tv) > 0.02 * abs(tv) + 0.05)
R["B4"]["annotated"] = dict(n_values=n, error_rate=e / n)
for m in MODELS:
    z = tf[(tf.model == m) & (tf.budget == 768)]; n = e = 0
    for vv, i in zip(z.vals, z.scenario_id):
        for b, (x, _) in vv.items():
            tv = S[i]["eu"][np.isclose(S[i]["bids"], b)][0]; n += 1; e += int(abs(x - tv) > 0.02 * abs(tv) + 0.05)
    R["B4"][m] = dict(n_values=n, off_by_more_than_2pct=e / n)

# ---------------- B5 difficulty
sc = pd.read_csv("../scenarios.csv").set_index("scenario_id")
y = tf[tf.budget.isin([768, -1]) & tf.complete].copy()
y["passed"] = (~np.isclose(y.chosen_bid, y.best_eval_bid)).astype(int)
y["gap12"] = y.scenario_id.map(lambda i: S[i]["gap12"]); y["product"] = y.scenario_id.map(lambda i: S[i]["product"])
y["window"] = y.scenario_id.map(lambda i: int(sc.loc[i, "candidate_seed"] % 2 == 0))
y["near_tie"] = (y.gap12 < 0.02).astype(int)
for m in MODELS:
    z = y[y.model == m]
    f = smf.logit("passed ~ gap12 + C(K) + C(product) + window + C(budget)", data=z).fit(disp=0)
    R["B5"][m] = dict(coef_gap12=float(f.params["gap12"]), p_gap12=float(f.pvalues["gap12"]),
                      coef_window=float(f.params["window"]), p_window=float(f.pvalues["window"]),
                      near_tie_share_passed=float(z[z.passed == 1].near_tie.mean()), near_tie_share_all=float(z.near_tie.mean()),
                      passed_rate_near_tie=float(z[z.near_tie == 1].passed.mean()) if (z.near_tie == 1).any() else None,
                      passed_rate_not_tie=float(z[z.near_tie == 0].passed.mean()),
                      chance_1_over_Km1=float((1 / (z[z.passed == 1].K - 1)).mean()))

# ---------------- B6 monitoring
for m in MODELS:
    z = tf[(tf.model == m) & (tf.budget == 768)]
    rb = [pick(vv, 2, b) for vv, b in zip(z.vals, z.chosen_bid)]
    flag = ~np.isclose(np.array(rb, float), z.chosen_bid.values)
    wrong = ~np.isclose(z.chosen_bid.values, z.calc.values)
    R["B6"][m] = dict(flag_rate=float(flag.mean()), precision=float(wrong[flag].mean()) if flag.any() else None,
                      recall=float(flag[wrong].mean()), reader_right_when_flag=float(np.isclose(np.array(rb, float)[flag], z.calc.values[flag]).mean()))

# ---------------- B7 economics
held = {}
for r in sc.reset_index().itertuples():
    c = json.loads(r.candidates_json); held[r.scenario_id] = {float(x["bid"]): x for x in c}
for m in MODELS:
    z = tf[(tf.model == m) & (tf.budget.isin([0, 768]))]
    out = {}
    for bud in (0, 768):
        w = z[z.budget == bud]
        wrong = w[~np.isclose(w.chosen_bid, w.calc)]
        over = (wrong.chosen_bid > wrong.calc).mean()
        dev = ((wrong.chosen_bid - wrong.calc) / wrong.scenario_id.map(lambda i: S[i]["v"])).mean()
        def tp(i, b): return held[i][float(b)]["test_p"]
        pay_m = np.mean([tp(i, b) * b for i, b in zip(w.scenario_id, w.chosen_bid)]); pay_c = np.mean([tp(i, S[i]["calc"]) * S[i]["calc"] for i in w.scenario_id])
        win_m = np.mean([tp(i, b) for i, b in zip(w.scenario_id, w.chosen_bid)]); win_c = np.mean([tp(i, S[i]["calc"]) for i in w.scenario_id])
        U = lambda i, b: held[i][float(b)]["test_u"]
        loss_rel = {p_: float(np.mean([(U(i, S[i]["calc"]) - U(i, b)) / S[i]["v"] for i, b in zip(w.scenario_id, w.chosen_bid) if S[i]["product"] == p_])) for p_ in PRODUCTS}
        loss_usd = {p_: float(np.mean([(U(i, S[i]["calc"]) - U(i, b)) for i, b in zip(w.scenario_id, w.chosen_bid) if S[i]["product"] == p_])) for p_ in PRODUCTS}
        out[str(bud)] = dict(wrong_share=float(len(wrong) / len(w)), overbid_share_of_wrong=float(over), mean_dev_over_v=float(dev),
                             pay_model=float(pay_m), pay_calc=float(pay_c), win_model=float(win_m), win_calc=float(win_c),
                             loss_over_v=loss_rel, loss_usd=loss_usd)
    R["B7"][m] = out

# ---------------- B8 intervals
tf["q_ge2"] = tf["q_ge2_600"]; tf["q_km1"] = tf["q_km1_600"]
for m in MODELS:
    z = tf[(tf.model == m) & (tf.budget == 768)].reset_index(drop=True)
    z0 = tf[(tf.model == m) & (tf.budget == 0)].set_index(["scenario_id", "repeat_seed"])
    gg = d[(d.model == m) & (d.decoding == "greedy")]
    U0 = gg[gg.budget == 0].set_index(["scenario_id", "repeat_seed"]).test_utility
    U7 = gg[gg.budget == 768].set_index(["scenario_id", "repeat_seed"]).test_utility
    z["U0"] = [U0.get((i, s_)) for i, s_ in zip(z.scenario_id, z.repeat_seed)]; z["U7"] = [U7.get((i, s_)) for i, s_ in zip(z.scenario_id, z.repeat_seed)]
    z["Uc"] = [held[i][S[i]["calc"]]["test_u"] for i in z.scenario_id]
    z["q0r"] = [z0.q_read.get((i, s_)) for i, s_ in zip(z.scenario_id, z.repeat_seed)]
    def stats_(w):
        base = w.q0r.mean(); den = w.qcalc.mean() - base
        return [(w[c].mean() - base) / den for c in ("q_read", "q_ge2", "q_km1", "q_be")] + [(w.U7.mean() - w.U0.mean()) / (w.Uc.mean() - w.U0.mean())]
    pt = stats_(z); bs = np.array([stats_(z.iloc[family_boot_ids(z.fam.values, rng)]) for _ in range(2000)])
    R["B8"][m] = {k: dict(pt=float(pt[j]), ci=[float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]) for j, k in enumerate(["answer", "ge2", "km1", "oracle", "dollar"])}

json.dump(R, open(f"{OUTDIR}/t10_judge7.json", "w"), indent=1, default=float)
print(json.dumps(R, default=lambda o: round(float(o), 3))[:200])

# ---------------- per-budget series for Figure 1c
FIG = {}
for m in MODELS:
    FIG[m] = [dict(budget=int(b_), q_read=float(z.q_read.mean()), q_reader=float(z.q_ge2_600.mean()), q_oracle=float(z.q_be.mean()))
              for b_, z in tf[(tf.model == m) & (tf.budget >= 0)].groupby("budget")]
json.dump(FIG, open(f"{OUTDIR}/t10_fig_series.json", "w"), indent=1)
