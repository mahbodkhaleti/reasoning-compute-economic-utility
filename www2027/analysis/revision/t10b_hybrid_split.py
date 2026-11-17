"""B2 (final): decomposition with hybrid stated values. An option's value is its true expected profit when the parser
finds that value in the trace (computed correctly), else the number the text reader reads for it (miscomputed or
misread), else the option is unwritten. Categories in order: Not reached (best unwritten), Arith. (the trace's values
do not put the best option first), Incomplete (not all options written), Passed (all written, best first, answer
differs). Also per-option miscompute rate vs annotators, and the split for unforced traces that stopped on their own."""
import json, numpy as np, pandas as pd
exec(open("t10_judge7.py").read().split("# ---------------- B2")[0].split("for W in (300, 600, 1000):\n    for nm")[0])
rng = np.random.default_rng(20261119)
def hybrid(vals, fp, i):
    s = S[i]; out = {}
    for b, (x, _) in vals.items(): out[b] = (x, "read")
    for x in fp:
        b = float(x); out[b] = (float(s["eu"][np.isclose(s["bids"], b)][0]), "true")
    return out
tf["hv"] = [hybrid(vv, fp, i) for vv, fp, i in zip(tf.vals, tf.fp, tf.scenario_id)]
def classify(hv, i, K, chosen):
    s = S[i]; calc = s["calc"]; eu = lambda o: s["eu"][np.isclose(s["bids"], o)][0]
    if np.isclose(chosen, calc): return "right"
    if calc not in hv: return "not_reached"
    if any(hv[o][0] >= hv[calc][0] for o in hv if o != calc and eu(o) < eu(calc)): return "arith"
    if len(hv) < K: return "incomplete"
    return "passed"
tf["cls"] = [classify(hv, i, K, b) for hv, i, K, b in zip(tf.hv, tf.scenario_id, tf.K, tf.chosen_bid)]
tf["gap"] = tf.qcalc - tf.q_read
cats = ["not_reached", "arith", "incomplete", "passed"]
def decompose(z):
    gap = z.gap.values; c = z.cls.values; fams = z.fam.values
    def sh(ids):
        T = gap[ids].sum(); return [gap[ids][c[ids] == k].sum() / T for k in cats]
    pt = sh(np.arange(len(z))); bs = np.array([sh(family_boot_ids(fams, rng)) for _ in range(2000)])
    return {k: dict(share=float(pt[j]), ci=[float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))], n=int((c == k).sum())) for j, k in enumerate(cats)}
R = {"forced768": {}, "natural_stop": {}, "miscompute_rate": {}, "complete_share_written": {}}
for m in MODELS:
    z = tf[(tf.model == m) & (tf.budget == 768)].reset_index(drop=True)
    R["forced768"][m] = decompose(z)
    R["forced768"][m]["missed_written_best"] = R["forced768"][m]["incomplete"]["share"] + R["forced768"][m]["passed"]["share"]
    zn = tf[(tf.model == m) & (tf.budget == -1) & (tf.reasoning_tokens < 1024)].reset_index(drop=True)
    R["natural_stop"][m] = dict(n=int(len(zn)), share_of_run=float(len(zn) / len(tf[(tf.model == m) & (tf.budget == -1)])),
                                q=float(zn.q_read.mean()), wrong=float((zn.cls != "right").mean()),
                                **({"split": decompose(zn)} if (zn.cls != "right").sum() >= 20 else {}))
    n_read = sum(sum(1 for b, (x, k) in hv.items() if k == "read") for hv in z.hv); n_all = sum(len(hv) for hv in z.hv)
    R["miscompute_rate"][m] = dict(values=n_all, not_matching_true=n_read / max(n_all, 1))
json.dump(R, open(f"{OUTDIR}/t10b_hybrid_split.json", "w"), indent=1, default=float)
f = lambda x: round(100 * x)
for m in MODELS:
    v = R["forced768"][m]; print(LAB[m], {k: (f(v[k]["share"]), [f(c) for c in v[k]["ci"]], v[k]["n"]) for k in cats}, "missed", f(v["missed_written_best"]), "misc", round(R["miscompute_rate"][m]["not_matching_true"], 3))
    nn = R["natural_stop"][m]; print("   natural", nn["n"], round(nn["share_of_run"], 2), round(nn["q"], 3), round(nn["wrong"], 3), {k: (f(x["share"]), [f(c) for c in x["ci"]], x["n"]) for k, x in nn.get("split", {}).items()})

# ---------------- interruption test with hybrid values (forced 768): best written and ranked first
from statsmodels.discrete.conditional_models import ConditionalLogit
import statsmodels.formula.api as smf
rows = []
for r in tf[tf.budget == 768].itertuples():
    hv = r.hv; c = r.calc
    if c not in hv or any(hv[o][0] >= hv[c][0] for o in hv if o != c): continue
    n = max(r.n_chars_seg, 1)
    pos = r.vals[c][1] if c in r.vals else r.fp.get(str(c), r.fp.get(str(int(c)) if c == int(c) else str(c), n))
    rows.append(dict(model=r.model, K=r.K, complete=int(len(hv) >= r.K), right=int(np.isclose(r.chosen_bid, c)), tok_after=(n - pos) / n * 768))
B3 = pd.DataFrame(rows)
inc = B3[B3.complete == 0].copy(); inc["bin"] = pd.cut(inc.tok_after, [-1e9, 64, 192, 1e9], labels=["<64", "64-192", ">192"])
lg = smf.logit("right ~ np.log1p(tok_after.clip(lower=0)) + C(model) + C(K)", data=inc).fit(disp=0)
R["interruption"] = dict(right_complete={m: float(B3[(B3.model == m) & (B3.complete == 1)].right.mean()) for m in MODELS},
                         right_incomplete={m: float(B3[(B3.model == m) & (B3.complete == 0)].right.mean()) for m in MODELS},
                         n_incomplete={m: int(((B3.model == m) & (B3.complete == 0)).sum()) for m in MODELS},
                         pooled_bins={str(k): dict(n=int(len(x)), right=float(x.right.mean())) for k, x in inc.groupby("bin", observed=True)},
                         logit_coef=float(lg.params.iloc[1]), logit_p=float(lg.pvalues.iloc[1]))
_t10 = open("t10_judge7.py").read(); exec(_t10[_t10.index("# synthetic"):_t10.index("sq0 =")])
# ---------------- synthetic: hybrid split at 768 and within-scenario position logit
sy["fp"] = sy.first_pos.map(json.loads)
def hyb_s(vals, fp, e):
    out = {b: (x, "read") for b, (x, _) in vals.items()}
    for x in fp: out[float(x)] = (e[float(x)], "true")
    return out
sy["hv"] = [hyb_s(vv, fp, e) for vv, fp, e in zip(sy.vals, sy.fp, sy.eu)]
def cls_s(hv, calc, e, K, ch):
    if np.isclose(ch, calc): return "right"
    if calc not in hv: return "not_reached"
    if any(hv[o][0] >= hv[calc][0] for o in hv if o != calc and e[o] < e[calc]): return "arith"
    return "incomplete" if len(hv) < K else "passed"
sy["cls"] = [cls_s(hv, c, e, K, b) for hv, c, e, K, b in zip(sy.hv, sy.calc, sy.eu, sy.K, sy.selected_bid)]
sy["gap"] = 1 - sy.q_read
R["synthetic768"] = {}; R["synthetic_position"] = {}
for m in MODELS:
    z = sy[(sy.model == m) & (sy.budget == 768)]
    T = z.gap.sum(); R["synthetic768"][m] = {k: float(z.gap[z.cls == k].sum() / T) for k in cats}
    R["synthetic768"][m]["missed_written_best"] = R["synthetic768"][m]["incomplete"] + R["synthetic768"][m]["passed"]
    w = z[z.fp.map(len) == z.K].copy()
    if len(w) == 0: continue
    w["order"] = w.disp.map(lambda L: [float(b) for b, p in L])
    w["best_e"] = [max(((float(x), e[float(x)]) for x in fp), key=lambda kv: kv[1])[0] for fp, e in zip(w.fp, w.eu)]
    w["ign"] = (~np.isclose(w.selected_bid, w.best_e)).astype(int)
    w["pos"] = [(o.index(b)) / (k - 1) for o, b, k in zip(w.order, w.best_e, w.K)]
    w["grp"] = w.scenario_id
    var = w.groupby("grp").ign.transform(lambda s: s.nunique() > 1); ww = w[var]
    if ww.ign.sum() >= 15 and len(ww) > 20:
        fit = ConditionalLogit(ww.ign.values, ww[["pos"]].values, groups=pd.factorize(ww.grp)[0]).fit(disp=0)
        ci = np.exp(fit.conf_int())
        R["synthetic_position"][m] = dict(or_=float(np.exp(fit.params[0])), ci=[float(ci[0, 0]), float(ci[0, 1])], p=float(fit.pvalues[0]), events=int(ww.ign.sum()))
    else:
        R["synthetic_position"][m] = dict(events=int(ww.ign.sum()))
json.dump(R, open(f"{OUTDIR}/t10b_hybrid_split.json", "w"), indent=1, default=float)
print(json.dumps(R["interruption"], default=lambda o: round(float(o), 3)))
for m in MODELS: print(LAB[m], {k: f(x) for k, x in R["synthetic768"][m].items()}, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in R["synthetic_position"][m].items()})
