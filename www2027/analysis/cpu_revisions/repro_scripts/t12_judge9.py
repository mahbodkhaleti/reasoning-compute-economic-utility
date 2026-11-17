"""Round-9 judge fixes (CPU, saved data only).
D1  Truncation-free subsets: forced-768 traces that finished the list by token 600; unforced traces that finished the
    list (any model); passed-over rate and share of their gap. 768-forced vs unforced q for Granite.
D2  Split the gap into truncation-driven (not reached + incomplete) vs passed over a finished list vs arithmetic.
D3  Audit on the synthetic traces (not used to design the reader); compare with always calling the calculator.
D4  Per-product gap split and q at 768 (family bootstrap within product).
D5  Second-price counterfactual (eBay's own rule): win if bid >= closing price, pay the closing price; re-score the
    saved choices; does high-bid drift still cost?
D6  DeepSeek oracle interval: held-out loss of the decisions where the oracle differs from the calculator.
D7  Intervals for: reader and answer at 512; voting; correct-when-stopping rate.
"""
import json, numpy as np, pandas as pd
_src = open("t11_judge8.py").read()
exec(_src[:_src.index("# ---------------- C1")])
R = {k: {} for k in ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]}
z768 = tf[tf.budget == 768]
for m in MODELS:
    z = z768[z768.model == m]
    early = z[z.complete & (z.comp_tok <= 600)]
    nat = tf[(tf.model == m) & (tf.budget == -1)]
    natc = nat[nat.vals.map(len) >= nat.K] if False else nat[nat.complete]
    def passed_rate(w):
        if not len(w): return None
        return float((~np.isclose(w.chosen_bid, w.best_eval_bid)).mean())
    R["D1"][m] = dict(n_early=int(len(early)), passed_rate_early=passed_rate(early), q_early=float(early.q_read.mean()) if len(early) else None,
                      qcalc_early=float(early.qcalc.mean()) if len(early) else None,
                      passed_rate_all_complete768=passed_rate(z[z.complete]),
                      n_nat_complete=int(len(natc)), passed_rate_nat_complete=passed_rate(natc),
                      q_nat_complete=float(natc.q_read.mean()), qcalc_nat_complete=float(natc.qcalc.mean()),
                      q_forced768=float(z.q_read.mean()), q_unforced=float(nat.q_read.mean()))
    T = z.gap.sum(); c = z.cls
    R["D2"][m] = dict(truncation=float(z.gap[c.isin(["not_reached", "incomplete"])].sum() / T), passed=float(z.gap[c == "passed"].sum() / T),
                      arith=float(z.gap[c == "arith"].sum() / T))
# D3 synthetic audit
_t10 = open("t10_judge7.py").read(); exec(_t10[_t10.index("# synthetic"):_t10.index("sq0 =")])
for m in MODELS:
    z = sy[(sy.model == m) & (sy.budget == 768)]
    rb = np.array([pick(vv, 2, b) for vv, b in zip(z.vals, z.selected_bid)], float)
    flag = ~np.isclose(rb, z.selected_bid.values); wrong = ~np.isclose(z.selected_bid.values, z.calc.values)
    qpol = np.where(flag, 1.0, z.q_read.values); q0s = sy[(sy.model == m) & (sy.budget == 0)].q_read.mean()
    R["D3"][m] = dict(flag_rate=float(flag.mean()), recall=float(flag[wrong].mean()) if wrong.any() else None, precision=float(wrong[flag].mean()) if flag.any() else None,
                      false_alarm=float(flag[~wrong].mean()), gap_recovered_policy=float((qpol.mean() - q0s) / (1 - q0s)),
                      gap_recovered_answer=float((z.q_read.mean() - q0s) / (1 - q0s)))
# D4 per product
for m in MODELS:
    z = z768[z768.model == m].copy(); z["product"] = z.scenario_id.map(lambda i: S[i]["product"])
    zz0 = tf[(tf.model == m) & (tf.budget == 0)].copy(); zz0["product"] = zz0.scenario_id.map(lambda i: S[i]["product"])
    out = {}
    for p in PRODUCTS:
        w = z[z["product"] == p]; T = w.gap.sum(); q0p = zz0[zz0["product"] == p].q_read.mean()
        out[p] = dict(q768=float(w.q_read.mean()), qcalc=float(w.qcalc.mean()), gap_rec=float((w.q_read.mean() - q0p) / (w.qcalc.mean() - q0p)),
                      split={k: float(w.gap[w.cls == k].sum() / T) for k in cats},
                      missed_written_best=float(w.gap[w.cls.isin(["incomplete", "passed"])].sum() / T))
    R["D4"][m] = out
# D5 second-price counterfactual
au = pd.read_csv("../auction_splits.csv")
def prod_of(item):
    it = item.lower(); return next(p for p in PRODUCTS if p.split()[0].lower() in it)
au["product"] = au["item"].map(prod_of)
test = {p: au[(au["product"] == p) & (au.split != "train")].final_price.values for p in PRODUCTS}
def U2(p, v, b):
    x = test[p]; return float(np.mean(np.where(x <= b, v - x, 0.0)))
for m in MODELS:
    out = {}
    for bud in (0, 768):
        z = tf[(tf.model == m) & (tf.budget == bud)]
        qs, qc, over_cost = [], [], []
        for i, b, c in zip(z.scenario_id, z.chosen_bid, z.calc):
            s = S[i]; U = {bb: U2(s["product"], s["v"], bb) for bb in s["bids"]}; um = max(U.values())
            qs.append(U[float(b)] / um); qc.append(U[s["calc"]] / um)
            if b > c: over_cost.append(U[float(b)] - U[c])
        out[str(bud)] = dict(q=float(np.mean(qs)), q_calc_choice=float(np.mean(qc)), overbid_gain_vs_calc=float(np.mean(over_cost)) if over_cost else None)
    R["D5"][m] = out
# D6
for m in ["deepseek_r1_llama_8b"]:
    z = z768[z768.model == m]; diff = ~np.isclose(z.best_eval_bid.fillna(-1), z.calc)
    R["D6"][m] = dict(n_diff=int(diff.sum()), q_be_diff=float(z.q_be[diff].mean()), qcalc_diff=float(z.qcalc[diff].mean()),
                      n_no_eval=int(z.best_eval_bid.isna().sum()))
# D7 intervals at 512 and correct-when-stopping
for m in MODELS:
    z = tf[(tf.model == m) & (tf.budget == 512)].reset_index(drop=True)
    z["q_r2"] = [q_of(i, pick(vv, 2, b)) for i, vv, b in zip(z.scenario_id, z.vals, z.chosen_bid)]
    def st(w): return [(w.q_read.mean() - q0[m]) / (w.qcalc.mean() - q0[m]), (w.q_r2.mean() - q0[m]) / (w.qcalc.mean() - q0[m])]
    bs = np.array([st(z.iloc[family_boot_ids(z.fam.values, rng)]) for _ in range(1000)])
    zn = tf[(tf.model == m) & (tf.budget == -1) & (tf.reasoning_tokens < 1024)]
    right = np.isclose(zn.chosen_bid, zn.calc)
    R["D7"][m] = dict(ans512=[float(x) for x in st(z)][0], ans512_ci=[float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))],
                      rd512=[float(x) for x in st(z)][1], rd512_ci=[float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))],
                      natural_stop_n=int(len(zn)), natural_stop_right=float(right.mean()))
json.dump(R, open(f"{OUTDIR}/t12_judge9.json", "w"), indent=1, default=float)
r3 = lambda o: round(float(o), 3)
for k in R:
    print("=====", k)
    for m, v in R[k].items(): print(m, json.dumps(v, default=r3))
