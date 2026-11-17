"""Round-11 judge fixes (CPU, saved data only).
F1  First-price equilibrium win curve: rivals draw values from the lognormal fitted in t13 (closing price = second-highest
    of n values) and bid the symmetric first-price equilibrium beta(x) = x - int_0^x G^{N-1} / G(x)^{N-1} with N = n+1
    bidders. A bid b wins with probability mean_j G(beta^{-1}(b))^{n_j}. Re-score choices and the calculator.
F2  Table 2 split and the prefix analysis by K.
F3  Two-sided: answer right although the written values rank another option first (share of decisions).
F4  Audit vs trivial triggers at matched call rate: unfinished list (reader count < K), K=7, low letter probability,
    random. Gap recovered and per-call gain.
F5  Dollar-weighted decomposition.
"""
import json, numpy as np, pandas as pd
from scipy import stats
_src = open("t13_judge10.py").read()
exec(_src[:_src.index("# ---------------- E2")].replace('for m in MODELS: R["E1"][m] = score_curve(OS, m)', ''))
R = {k: {} for k in ["F1", "F2", "F3", "F4", "F5"]}
fit = R_E1 = None
# ---------------- F1
FP = {}
for p, h in ho.items():
    h2 = h[h.n_bidders >= 2]; mu, sd = fit_G(h2.final_price.values, h2.n_bidders.values.astype(float))
    xs = np.exp(np.linspace(mu - 5 * sd, mu + 5 * sd, 4000)); G = stats.norm.cdf((np.log(xs) - mu) / sd)
    curves = {}
    for n in np.unique(h.n_bidders.clip(lower=1).values):
        N = n + 1; GN = G ** (N - 1)
        integ = np.concatenate([[0], np.cumsum(0.5 * (GN[1:] + GN[:-1]) * np.diff(xs))])
        beta = xs - np.where(GN > 1e-300, integ / np.maximum(GN, 1e-300), 0)
        beta = np.maximum.accumulate(beta)
        curves[n] = (beta, G ** n)
    ns = h.n_bidders.clip(lower=1).values
    def f(b, curves=curves, ns=ns):
        return float(np.mean([np.interp(b, curves[n][0], curves[n][1]) for n in ns]))
    FP[p] = f
for m in MODELS: R["F1"][m] = score_curve(FP, m)
# ---------------- F2
z768 = tf[tf.budget == 768]
for m in MODELS:
    for K in (5, 7):
        z = z768[(z768.model == m) & (z768.K == K)]; T = z.gap.sum()
        R["F2"].setdefault(m, {})[f"K{K}"] = dict(split={k: float(z.gap[z.cls == k].sum() / T) for k in cats}, complete=float(z.complete.mean()))
k512 = tf[tf.budget == 512].set_index(["model", "scenario_id", "repeat_seed"])
for m in MODELS:
    for K in (5, 7):
        rows = []
        for r in z768[(z768.model == m) & (z768.K == K)].itertuples():
            a = k512.loc[(m, r.scenario_id, r.repeat_seed)]; hv5 = a.hv; c = r.calc
            if c not in hv5 or any(hv5[o][0] >= hv5[c][0] for o in hv5 if o != c) or len(hv5) >= K: continue
            rows.append(dict(r5=int(np.isclose(a.chosen_bid, c)), r7=int(np.isclose(r.chosen_bid, c)), comp=int(len(r.hv) >= K)))
        x = pd.DataFrame(rows)
        R["F2"][m][f"prefix_K{K}"] = dict(n=int(len(x)), right512=float(x.r5.mean()) if len(x) else None,
                                         right768_complete=float(x[x.comp == 1].r7.mean()) if (x.comp == 1).sum() else None, n_complete=int(x.comp.sum()) if len(x) else 0)
# ---------------- F3
for m in MODELS:
    z = z768[z768.model == m]
    right = np.isclose(z.chosen_bid, z.calc)
    bad_rank = np.array([(c in hv) and any(hv[o][0] >= hv[c][0] for o in hv if o != c) for hv, c in zip(z.hv, z.calc)])
    top_other = np.array([len(hv) > 0 and not np.isclose(max(hv.items(), key=lambda kv: kv[1][0])[0], c) for hv, c in zip(z.hv, z.calc)])
    R["F3"][m] = dict(right_despite_trace_rank=float((right & top_other).mean()), right_share=float(right.mean()),
                      right_with_calc_written_not_first=float((right & bad_rank).mean()))
# ---------------- F4
for m in MODELS:
    z = z768[z768.model == m].reset_index(drop=True)
    rb = np.array([pick(vv, 2, b) for vv, b in zip(z.vals, z.chosen_bid)], float)
    flags = {"audit": ~np.isclose(rb, z.chosen_bid.values), "unfinished": (z.vals.map(len) < z.K).values, "K7": (z.K == 7).values}
    rate = flags["audit"].mean()
    pr = z.constrained_option_probability.values if "constrained_option_probability" in z else None
    if pr is None:
        dd = d[(d.model == m) & (d.decoding == "greedy") & (d.budget == 768)].set_index(["scenario_id", "repeat_seed"]).constrained_option_probability
        pr = np.array([dd.get((i, s_)) for i, s_ in zip(z.scenario_id, z.repeat_seed)], float)
    tau = np.quantile(pr, rate); flags["low_letter_prob"] = pr <= tau
    rr = np.random.default_rng(1); flags["random_same_rate"] = rr.random(len(z)) < rate
    out = {}
    for k, fl in flags.items():
        qp = np.where(fl, z.qcalc.values, z.q_read.values)
        out[k] = dict(call_rate=float(fl.mean()), gap_recovered=float((qp.mean() - q0[m]) / (z.qcalc.mean() - q0[m])),
                      gain_per_call=float((qp.mean() - z.q_read.mean()) / max(fl.mean(), 1e-9)))
    R["F4"][m] = out
# ---------------- F5
for m in MODELS:
    z = z768[z768.model == m]
    du = np.array([held_u(i, c) - held_u(i, b) for i, b, c in zip(z.scenario_id, z.chosen_bid, z.calc)]) if False else None
    hd = {r.scenario_id: {float(x["bid"]): x for x in json.loads(r.candidates_json)} for r in sc.reset_index().itertuples()} if m == MODELS[0] else hd
    du = np.array([hd[i][c]["test_u"] - hd[i][float(b)]["test_u"] for i, b, c in zip(z.scenario_id, z.chosen_bid, z.calc)])
    T = du.sum(); R["F5"][m] = {k: float(du[z.cls.values == k].sum() / T) for k in cats}
json.dump(R, open(f"{OUTDIR}/t14_judge11.json", "w"), indent=1, default=float)
r3 = lambda o: round(float(o), 3)
for k in R:
    print("=====", k)
    for m, v in R[k].items(): print(m, json.dumps(v, default=r3))
