"""Item 4: the audit (send the decision to the calculator when the text reader's pick differs from the answer),
evaluated out of sample.
(a) Reader settings (window 300/600/1000 characters x needs 2 or K-1 written values) chosen on synthetic traces only,
    tested on eBay traces, and the reverse. Selection criterion as in the paper: mean over models of the reader's
    improvement over the answer at 768 tokens.
(b) Nested cross-validation on eBay: outer folds = leave one product out, and separately 6 folds of value families
    (3 per product... families assigned to folds at random within product); settings chosen on the training folds only.
(c) Gap recovered against call rate (eBay, 768 tokens, mean over the six models): the audit (ranked by how far the
    answered option's written value falls below the largest written value), random calls, lowest letter probability
    first, unfinished list first (fewest written values first), at every call rate."""
import sys; sys.path.insert(0, "/tmp/www2027/analysis/rev25")
from r_common import *
_t10 = open("/tmp/www2027/analysis/revision/t10_judge7.py").read()
exec(_t10[_t10.index("# synthetic"):_t10.index("sq0 =")].replace("../", "/tmp/www2027/analysis/"))
SET = [(W, n) for W in (300, 600, 1000) for n in ("2", "K-1")]
for W in (300, 1000):
    sy[f"vals{W}"] = [read_vals(json.loads(sg) if isinstance(sg, str) else [], v, inf, W) for sg, v, inf in zip(sy.segments, sy.v, sy.pinfo)]
sy["vals600"] = sy["vals"]
need = lambda n, K: (K - 1) if n == "K-1" else 2
# eBay pieces
E = tf[tf.budget.isin([0, 768])].copy(); E["product"] = E.scenario_id.map(lambda i: S[i]["product"])
pr = g[g.budget == 768].set_index(["model", "scenario_id", "repeat_seed"]).constrained_option_probability
E7 = E[E.budget == 768].reset_index(drop=True)
E7["letter_p"] = [pr.get((m, i, s_), np.nan) for m, i, s_ in zip(E7.model, E7.scenario_id, E7.repeat_seed)]
E0 = E[E.budget == 0]
def e_reader_q(z, W, n): return np.array([q_of(i, pick(vv, need(n, K), b)) for i, vv, K, b in zip(z.scenario_id, z[f"vals{W}"], z.K, z.chosen_bid)])
def e_flag(z, W, n): return np.array([not np.isclose(pick(vv, need(n, K), b), b) for vv, K, b in zip(z[f"vals{W}"], z.K, z.chosen_bid)])
def s_reader_q(z, W, n): return np.array([e[float(pick(vv, need(n, K), b))] / u for e, vv, K, b, u in zip(z.eu, z[f"vals{W}"], z.K, z.selected_bid, z.umax)])
def s_flag(z, W, n): return np.array([not np.isclose(pick(vv, need(n, K), b), b) for vv, K, b in zip(z[f"vals{W}"], z.K, z.selected_bid)])
S7 = sy[sy.budget == 768].reset_index(drop=True); S0 = sy[sy.budget == 0]
sq0m = S0.groupby("model").q_read.mean()
def choose_ebay(mask):
    sc = {f"{W}|{n}": float(np.mean([(e_reader_q(E7[mask & (E7.model == m)], W, n).mean() - E7[mask & (E7.model == m)].q_read.mean()) for m in MODELS])) for W, n in SET}
    return max(sc, key=sc.get), sc
def choose_syn():
    sc = {f"{W}|{n}": float(np.mean([(s_reader_q(S7[S7.model == m], W, n).mean() - S7[S7.model == m].q_read.mean()) for m in MODELS])) for W, n in SET}
    return max(sc, key=sc.get), sc
def ebay_audit(z, z0, W, n):
    fl = e_flag(z, W, n); qp = np.where(fl, z.qcalc.values, z.q_read.values); q0_ = z0.q_read.mean()
    return dict(gap_audit=float((qp.mean() - q0_) / (z.qcalc.mean() - q0_)), gap_answer=float((z.q_read.mean() - q0_) / (z.qcalc.mean() - q0_)), calls=float(fl.mean()))
def syn_audit(z, z0, W, n):
    fl = s_flag(z, W, n); qp = np.where(fl, 1.0, z.q_read.values); q0_ = z0.q_read.mean()
    return dict(gap_audit=float((qp.mean() - q0_) / (1 - q0_)), gap_answer=float((z.q_read.mean() - q0_) / (1 - q0_)), calls=float(fl.mean()))
R = {}
# (a) cross-dataset
cs, scs = choose_syn(); ce, sce = choose_ebay(np.ones(len(E7), bool))
R["chosen_on_synthetic"] = cs; R["chosen_on_ebay"] = ce; R["scores_syn"] = scs; R["scores_ebay"] = sce
W, n = int(cs.split("|")[0]), cs.split("|")[1]
R["syn_to_ebay"] = {m: ebay_audit(E7[E7.model == m], E0[E0.model == m], W, n) for m in MODELS}
W, n = int(ce.split("|")[0]), ce.split("|")[1]
R["ebay_to_syn"] = {m: syn_audit(S7[S7.model == m], S0[S0.model == m], W, n) for m in MODELS}
# (b) nested CV on eBay
def nested(fold_of):
    folds = sorted(set(fold_of)); qp_all = {m: np.zeros(int((E7.model == m).sum())) for m in MODELS}; calls = {m: 0 for m in MODELS}; chosen = []
    for f in folds:
        tr = np.array([x != f for x in fold_of]); best, _ = choose_ebay(tr); chosen.append(best)
        W, n = int(best.split("|")[0]), best.split("|")[1]
        for m in MODELS:
            mm = (E7.model == m).values; z = E7[mm & ~tr]; fl = e_flag(z, W, n)
            idx = np.where(~tr[mm])[0]; qp_all[m][idx] = np.where(fl, z.qcalc.values, z.q_read.values); calls[m] += fl.sum()
    out = {}
    for m in MODELS:
        z = E7[E7.model == m]; q0_ = E0[E0.model == m].q_read.mean()
        out[m] = dict(gap_audit=float((qp_all[m].mean() - q0_) / (z.qcalc.mean() - q0_)), calls=float(calls[m] / len(z)))
    return out, chosen
R["nested_leave_product_out"], R["nested_lpo_chosen"] = nested(E7["product"].values)
rng_ = np.random.default_rng(5); fam_fold = {}
for p in PRODUCTS:
    fs = list(FAM_BY_PROD[p]); rng_.shuffle(fs)
    for k, f in enumerate(fs): fam_fold[f] = k % 6
R["nested_family_6fold"], R["nested_fam_chosen"] = nested(E7.fam.map(fam_fold).values)
# (c) curves on eBay at 768 with the paper's setting (600, 2)
grid = np.linspace(0, 1, 101); curves = {k: [] for k in ("audit", "random", "low_letter_prob", "unfinished_list")}
for m in MODELS:
    z = E7[E7.model == m].reset_index(drop=True); q0_ = E0[E0.model == m].q_read.mean(); N = len(z)
    gapf = lambda order: np.array([(np.where(np.isin(np.arange(N), order[:int(round(r * N))]), z.qcalc.values, z.q_read.values).mean() - q0_) / (z.qcalc.mean() - q0_) for r in grid])
    rr = np.random.default_rng(11).random(N) * 1e-6
    def margin(vv, b):
        if len(vv) < 2: return -np.inf
        top = max(x for x, _ in vv.values()); mine = vv.get(float(b), (None,))[0]
        if np.isclose(pick(vv, 2, b), b): return -np.inf
        return (top - (mine if mine is not None else -1e9)) / max(abs(top), 1e-9)
    mg = np.array([margin(vv, b) for vv, b in zip(z.vals, z.chosen_bid)])
    # audit: flagged decisions first (largest margin first), then the rest in random order
    curves["audit"].append(gapf(np.lexsort((rr, -np.nan_to_num(mg, neginf=-1e18)))))
    curves["random"].append(np.mean([gapf(np.random.default_rng(s_).permutation(N)) for s_ in range(20)], axis=0))
    curves["low_letter_prob"].append(gapf(np.argsort(z.letter_p.values + rr)))
    curves["unfinished_list"].append(gapf(np.argsort(z.vals.map(len).values / z.K.values + rr)))
    R.setdefault("audit_flag_rate", {})[m] = float(np.isfinite(mg).mean())
R["curves"] = {k: dict(mean=np.mean(v, axis=0).tolist(), min=np.min(v, axis=0).tolist(), max=np.max(v, axis=0).tolist()) for k, v in curves.items()}
R["curve_grid"] = grid.tolist()
for k in curves:
    at = {r: float(np.interp(r, grid, np.mean(curves[k], axis=0))) for r in (0.1, 0.2, 0.3)}
    R.setdefault("curve_at", {})[k] = at
dump25("r4_audit_oos", R)
f = lambda x: f"{100*x:.0f}"
print("chosen on synthetic:", cs, "| on eBay:", ce)
for m in MODELS:
    a, b, c, d_ = R["syn_to_ebay"][m], R["ebay_to_syn"][m], R["nested_leave_product_out"][m], R["nested_family_6fold"][m]
    print(LAB[m], "syn->eBay audit", f(a["gap_audit"]), "(answer", f(a["gap_answer"]), "calls", f(a["calls"]), ") | eBay->syn audit", f(b["gap_audit"]), "(answer", f(b["gap_answer"]), "calls", f(b["calls"]),
          ") | nested LPO", f(c["gap_audit"]), "calls", f(c["calls"]), "| nested fam", f(d_["gap_audit"]), "calls", f(d_["calls"]))
print("LPO chosen", R["nested_lpo_chosen"], "fam chosen", R["nested_fam_chosen"])
print("curve at 10/20/30% calls:", {k: {r: f(v) for r, v in d.items()} for k, d in R["curve_at"].items()})
print("flag rates", {LAB[m]: f(v) for m, v in R["audit_flag_rate"].items()})
