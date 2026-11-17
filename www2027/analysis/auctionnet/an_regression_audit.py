"""AuctionNet: (1) pass-over regression with r5's formula, category in place of product; (2) the audit with the reader
setting chosen on eBay only (r4 'chosen_on_ebay', no AuctionNet data used), plus nested leave-one-category-out;
(3) gap recovered against call rate at 20% (audit, random, low letter probability, unfinished list).
Reads results/AN/an_traces[_closed].pkl from an_split.py. Usage: python an_regression_audit.py <results_dir> [closed]"""
import sys, json, re, numpy as np, pandas as pd, statsmodels.formula.api as smf
from an_common import build_S
B = sys.argv[1]; TAG = "_closed" if len(sys.argv) > 2 else ""
S = build_S(B); A = "/tmp/www2027/analysis"
LAB = {"deepseek_r1_llama_8b": "DeepSeek-8B", "phi4_mini_reasoning": "Phi-mini", "qwen3_4b": "Qwen3-4B", "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B", "granite_3_2_8b_instruct": "Granite-8B"}
MODELS = list(LAB)
def _grab(path, name):
    src = open(path).read(); i = src.index(f"def {name}("); j = re.search(r"\n(?=\S)", src[i + 1:]); return src[i: i + 1 + j.start()]
ns = dict(np=np); exec(_grab(f"{A}/revision/t10_judge7.py", "read_vals"), ns); exec(_grab(f"{A}/revision/t10_judge7.py", "pick"), ns)
read_vals, pick = ns["read_vals"], ns["pick"]
tf = pd.read_pickle(f"/tmp/www2027/results/AN/an_traces{TAG}.pkl")
q_of = lambda i, b: float(S[i]["tu"][np.isclose(S[i]["bids"], b)][0] / S[i]["ustar"])
# q(0) per model from the forced c0 decisions
q0 = {}
for m in MODELS:
    D = [json.loads(l) for l in open(f"{B}/auctionnet/decisions/{m}_forced.jsonl")]
    q0[m] = float(np.mean([r["test_utility_ratio"] for r in D if r["budget"] == 0]))
    lp = {(int(r["scenario_id"]), r["order_id"]): r["option_probabilities"][r["chosen_label"]] for r in D if r["budget"] == 768}
    z = (tf.model == m) & (tf.budget == 768)
    tf.loc[z, "letter_p"] = [lp[(i, s)] for i, s in zip(tf.loc[z, "scenario_id"], tf.loc[z, "repeat_seed"])]
R = {}
# (1) regression (r5 formula)
z = tf[tf.budget == 768].copy()
z = z[z.cls.isin(["incomplete", "passed", "right"])]
z = z[[(c in hv) and not any(hv[o][0] >= hv[c][0] for o in hv if o != c) for hv, c in zip(z.hv, z.calc)]].copy()
med = {p: np.median([S[i]["v"] for i in S if S[i]["product"] == p]) for p in {s["product"] for s in S.values()}}
z["passed_over"] = (~np.isclose(z.chosen_bid, z.calc)).astype(int)
z["gap12"] = z.scenario_id.map(lambda i: S[i]["gap12"])
opt = {}
for m in MODELS:
    for l in open(f"{B}/auctionnet/decisions/{m}_forced.jsonl"):
        r = json.loads(l)
        if r["budget"] == 768: opt[(m, int(r["scenario_id"]), r["order_id"])] = [float(b) for b in r["option_order"]]
z["order_list"] = [opt[(m, i, s)] for m, i, s in zip(z.model, z.scenario_id, z.repeat_seed)]
z["pos"] = [o.index(c) / (len(o) - 1) for o, c in zip(z.order_list, z.calc)]
z["K7"] = (z.K == 7).astype(int)
z["logv"] = [np.log(S[i]["v"] / med[S[i]["product"]]) for i in z.scenario_id]
z["product"] = z.scenario_id.map(lambda i: S[i]["product"].split()[0])
z["best_is_high"] = [int(np.isclose(S[i]["calc"], S[i]["bids"].max())) for i in z.scenario_id]
z["finished"] = [int(len(hv) >= K) for hv, K in zip(z.hv, z.K)]
z["mod"] = z.model.map(LAB); z["sid"] = pd.factorize(z.scenario_id)[0]
form = "passed_over ~ gap12 + pos + K7 + logv + C(product) + best_is_high + finished + C(mod, Treatment('Qwen3-8B'))"
fit = smf.logit(form, data=z).fit(disp=0, cov_type="cluster", cov_kwds={"groups": z.sid})
R["regression"] = {"n": int(len(z)), "events": int(z.passed_over.sum()), "share_best_is_high": float(z.best_is_high.mean()), "pooled": {}}
for k in fit.params.index:
    lo, hi = fit.conf_int().loc[k]; R["regression"]["pooled"][k] = dict(OR=float(np.exp(fit.params[k])), ci=[float(np.exp(lo)), float(np.exp(hi))], p=float(fit.pvalues[k]))
# (2) audit
SET = [(W, n) for W in (300, 600, 1000) for n in ("2", "K-1")]
need = lambda n, K: (K - 1) if n == "K-1" else 2
info = {i: dict(zip(S[i]["bids"], S[i]["dp"])) for i in S}
E7 = tf[tf.budget == 768].reset_index(drop=True)

ebay = json.load(open("/tmp/www2027/results/rev25/r4_audit_oos.json"))
We, ne = int(ebay["chosen_on_ebay"].split("|")[0]), ebay["chosen_on_ebay"].split("|")[1]
def flag(z, W, n): return np.array([not np.isclose(pick(vv, need(n, K), b), b) for vv, K, b in zip(z[f"vals{W}"], z.K, z.chosen_bid)])
def reader_q(z, W, n): return np.array([q_of(i, pick(vv, need(n, K), b)) for i, vv, K, b in zip(z.scenario_id, z[f"vals{W}"], z.K, z.chosen_bid)])
for W in (300, 600, 1000):
    E7[f"vals{W}"] = [read_vals(json.loads(sg) if isinstance(sg, str) else [], S[i]["v"], info[i], W) for sg, i in zip(E7.segments_raw, E7.scenario_id)]
def audit(z, m, W, n):
    fl = flag(z, W, n); qp = np.where(fl, z.qcalc.values, z.q_read.values)
    return dict(gap_audit=float((qp.mean() - q0[m]) / (z.qcalc.mean() - q0[m])), gap_answer=float((z.q_read.mean() - q0[m]) / (z.qcalc.mean() - q0[m])), calls=float(fl.mean()))
R["audit_ebay_setting"] = dict(setting=f"{We}|{ne}", by_model={LAB[m]: audit(E7[E7.model == m], m, We, ne) for m in MODELS})
def choose(mask):
    sc = {f"{W}|{n}": float(np.mean([reader_q(E7[mask & (E7.model == m)], W, n).mean() - E7[mask & (E7.model == m)].q_read.mean() for m in MODELS])) for W, n in SET}
    return max(sc, key=sc.get)
cat = E7.cat.values; out = {m: np.zeros(int((E7.model == m).sum())) for m in MODELS}; calls = {m: 0 for m in MODELS}; chosen = []
for c in sorted(set(cat)):
    tr = cat != c; best = choose(tr); chosen.append(best); W, n = int(best.split("|")[0]), best.split("|")[1]
    for m in MODELS:
        mm = (E7.model == m).values; z = E7[mm & ~tr]; fl = flag(z, W, n)
        out[m][np.where(~tr[mm])[0]] = np.where(fl, z.qcalc.values, z.q_read.values); calls[m] += fl.sum()
R["audit_nested_loco"] = dict(chosen=chosen, by_model={LAB[m]: dict(gap_audit=float((out[m].mean() - q0[m]) / (E7[E7.model == m].qcalc.mean() - q0[m])), calls=float(calls[m] / (E7.model == m).sum())) for m in MODELS})
# (3) curve at 20%
grid = np.linspace(0, 1, 101); curves = {k: [] for k in ("audit", "random", "low_letter_prob", "unfinished_list")}
for m in MODELS:
    z = E7[E7.model == m].reset_index(drop=True); N = len(z)
    gapf = lambda order: np.array([(np.where(np.isin(np.arange(N), order[:int(round(r * N))]), z.qcalc.values, z.q_read.values).mean() - q0[m]) / (z.qcalc.mean() - q0[m]) for r in grid])
    rr = np.random.default_rng(11).random(N) * 1e-6
    def margin(vv, b):
        if len(vv) < 2: return -np.inf
        top = max(x for x, _ in vv.values()); mine = vv.get(float(b), (None,))[0]
        if np.isclose(pick(vv, 2, b), b): return -np.inf
        return (top - (mine if mine is not None else -1e9)) / max(abs(top), 1e-9)
    mg = np.array([margin(vv, b) for vv, b in zip(z.vals600, z.chosen_bid)])
    curves["audit"].append(gapf(np.lexsort((rr, -np.nan_to_num(mg, neginf=-1e18)))))
    curves["random"].append(np.mean([gapf(np.random.default_rng(s_).permutation(N)) for s_ in range(20)], axis=0))
    curves["low_letter_prob"].append(gapf(np.argsort(z.letter_p.values + rr)))
    curves["unfinished_list"].append(gapf(np.argsort(z.vals600.map(len).values / z.K.values + rr)))
R["curve_at20"] = {k: float(np.interp(0.2, grid, np.mean(v, axis=0))) for k, v in curves.items()}
json.dump(R, open(f"/tmp/www2027/results/AN/an_regression_audit{TAG}.json", "w"), indent=1)
r = R["regression"]; print("regression n", r["n"], "events", r["events"], "share best_is_high", round(r["share_best_is_high"], 3))
for k in ("best_is_high", "finished", "pos", "gap12", "K7"): v = r["pooled"][k]; print(f"  {k} OR {v['OR']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]")
print("audit, eBay setting", R["audit_ebay_setting"]["setting"], {k: (round(100*v['gap_audit']), round(100*v['gap_answer']), round(100*v['calls'])) for k, v in R["audit_ebay_setting"]["by_model"].items()})
print("audit, nested LOCO", R["audit_nested_loco"]["chosen"], {k: (round(100*v['gap_audit']), round(100*v['calls'])) for k, v in R["audit_nested_loco"]["by_model"].items()})
print("at 20%", {k: round(100*v) for k, v in R["curve_at20"].items()})
