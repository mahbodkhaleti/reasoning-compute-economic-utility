"""Round-10 judge fixes (CPU, saved data only).
E1  Order-statistic win curve: under independent private values and proxy bidding, an auction's closing price is the
    second-highest of its n bidders' values. Fit a lognormal value distribution G per product by maximum likelihood on
    held-out (closing price, n) pairs with n >= 2; a new bidder wins at b with probability mean_j G(b)^{n_j}. Re-score
    the saved choices: gap recovered and overbid share of wrong answers.
E2  Economic layer per 1,000 decisions at 768 tokens vs the calculator's choice: seller revenue and bidder surplus
    under pay-your-bid (empirical curve, order-statistic curve) and under eBay's rule (pay the closing price).
E3  Reader leakage: choose the reader's window and threshold on Palm only (best gap recovered over all models), then
    report reader and audit results on Cartier and Xbox.
E4  Decomposition order: Incompl. assigned before Arith.
E5  Baselines that keep each model's own propensities: does a wrong 768 answer repeat the budget-0 answer for the
    same scenario under the same order, under another order (same bid), and how often would the bid-rank of a
    budget-0 answer from another scenario match (rank-preserving baseline)?
E6  Within-trace extension (greedy forced traces at 512 are prefixes of those at 768): for traces whose 512-token
    state has the best option written and ranked first but the list unfinished, answer accuracy at 512 vs at 768 by
    the 768 state.
E7  Stochastic seeds: q by budget (mean and seed SD), 5-seed majority vote at 768 vs one sample; passed-over rate is
    not available for sampled traces (not saved); energy per decision and per point of q recovered.
"""
import json, numpy as np, pandas as pd
from scipy import stats, optimize
_src = open("t11_judge8.py").read()
exec(_src[:_src.index("# ---------------- C1")])
R = {k: {} for k in ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]}
au = pd.read_csv("../auction_splits.csv")
def prod_of(item):
    it = item.lower(); return next(p for p in PRODUCTS if p.split()[0].lower() in it)
au["product"] = au["item"].map(prod_of)
ho = {p: au[(au["product"] == p) & (au.split != "train")] for p in PRODUCTS}
# ---------------- E1
def fit_G(c, n):
    lc = np.log(c)
    def nll(th):
        mu, ls = th; sd = np.exp(ls); z = (lc - mu) / sd
        G = np.clip(stats.norm.cdf(z), 1e-12, 1 - 1e-12); lg = stats.norm.logpdf(z) - np.log(sd) - lc
        return -np.sum(np.log(n * (n - 1)) + (n - 2) * np.log(G) + np.log(1 - G) + lg)
    r = optimize.minimize(nll, [lc.mean(), np.log(lc.std())], method="Nelder-Mead")
    return r.x[0], np.exp(r.x[1])
OS = {}
for p, h in ho.items():
    h2 = h[h.n_bidders >= 2]
    mu, sd = fit_G(h2.final_price.values, h2.n_bidders.values.astype(float))
    ns = h.n_bidders.clip(lower=1).values
    OS[p] = (lambda b, mu=mu, sd=sd, ns=ns: float(np.mean(stats.norm.cdf((np.log(b) - mu) / sd) ** ns)))
    R["E1"].setdefault("fit", {})[p] = dict(mu=float(mu), sd=float(sd), n_fit=int(len(h2)))
def score_curve(f, m):
    out = {}
    for bud in (0, 768):
        z = tf[(tf.model == m) & (tf.budget == bud)]
        qs, qc, over = [], [], []
        for i, b in zip(z.scenario_id, z.chosen_bid):
            s = S[i]; F = f[s["product"]]; U = {bb: F(bb) * (s["v"] - bb) for bb in s["bids"]}; um = max(U.values()); opt = max(U, key=U.get)
            qs.append(U[float(b)] / um if um > 0 else 1); qc.append(U[s["calc"]] / um if um > 0 else 1)
            if not np.isclose(b, opt): over.append(b > opt)
        out[str(bud)] = dict(q=float(np.mean(qs)), qcalc=float(np.mean(qc)), overbid_of_wrong=float(np.mean(over)))
    out["gap_recovered"] = (out["768"]["q"] - out["0"]["q"]) / (out["768"]["qcalc"] - out["0"]["q"])
    return out
for m in MODELS: R["E1"][m] = score_curve(OS, m)
# ---------------- E2
held = {r.scenario_id: {float(x["bid"]): x for x in json.loads(r.candidates_json)} for r in sc.reset_index().itertuples()}
for m in MODELS:
    z = tf[(tf.model == m) & (tf.budget == 768)]
    acc = {k: [0.0, 0.0] for k in ["pyb_emp_model", "pyb_emp_calc", "pyb_os_model", "pyb_os_calc", "sp_model", "sp_calc"]}
    for i, b in zip(z.scenario_id, z.chosen_bid):
        s = S[i]; x = ho[s["product"]].final_price.values; v = s["v"]
        for who, bb in (("model", float(b)), ("calc", s["calc"])):
            pe = held[i][bb]["test_p"]; acc[f"pyb_emp_{who}"][0] += pe * bb; acc[f"pyb_emp_{who}"][1] += pe * (v - bb)
            po = OS[s["product"]](bb); acc[f"pyb_os_{who}"][0] += po * bb; acc[f"pyb_os_{who}"][1] += po * (v - bb)
            win = x <= bb; acc[f"sp_{who}"][0] += np.mean(np.where(win, x, 0)); acc[f"sp_{who}"][1] += np.mean(np.where(win, v - x, 0))
    n = len(z)
    R["E2"][m] = {k: dict(revenue_per_1000=1000 * a / n, surplus_per_1000=1000 * b_ / n) for k, (a, b_) in acc.items()}
# ---------------- E3
tf["product"] = tf.scenario_id.map(lambda i: S[i]["product"])
grid = [(W, need) for W in (300, 600, 1000) for need in ("2", "K-1")]
def reader_q(z, W, need):
    return np.array([q_of(i, pick(vv, (K - 1) if need == "K-1" else 2, b)) for i, vv, K, b in zip(z.scenario_id, z[f"vals{W}"], z.K, z.chosen_bid)])
pal = tf[(tf.budget == 768) & (tf["product"] == "Palm Pilot M515 PDA")]
score = {f"{W}|{n}": float(np.mean([(reader_q(pal[pal.model == m], W, n).mean() - pal[pal.model == m].q_read.mean()) for m in MODELS])) for W, n in grid}
best = max(score, key=score.get); Wb, nb = int(best.split("|")[0]), best.split("|")[1]
R["E3"]["palm_scores"] = score; R["E3"]["chosen"] = best
for m in MODELS:
    z = tf[(tf.model == m) & (tf.budget == 768) & (tf["product"] != "Palm Pilot M515 PDA")]
    zz0 = tf[(tf.model == m) & (tf.budget == 0) & (tf["product"] != "Palm Pilot M515 PDA")]
    q0p = zz0.q_read.mean()
    qr = reader_q(z, Wb, nb)
    rb = np.array([pick(vv, (K - 1) if nb == "K-1" else 2, b) for vv, K, b in zip(z[f"vals{Wb}"], z.K, z.chosen_bid)], float)
    flag = ~np.isclose(rb, z.chosen_bid.values); wrong = ~np.isclose(z.chosen_bid.values, z.calc.values)
    qpol = np.where(flag, z.qcalc.values, z.q_read.values)
    R["E3"][m] = dict(answer=float((z.q_read.mean() - q0p) / (z.qcalc.mean() - q0p)), reader=float((qr.mean() - q0p) / (z.qcalc.mean() - q0p)),
                      policy=float((qpol.mean() - q0p) / (z.qcalc.mean() - q0p)), calls=float(flag.mean()), recall=float(flag[wrong].mean()),
                      false_alarm=float(flag[~wrong].mean()))
# ---------------- E4
def cls_alt(hv, i, K, chosen):
    s = S[i]; calc = s["calc"]; eu = lambda o: s["eu"][np.isclose(s["bids"], o)][0]
    if np.isclose(chosen, calc): return "right"
    if calc not in hv: return "not_reached"
    if len(hv) < K: return "incomplete"
    if any(hv[o][0] >= hv[calc][0] for o in hv if o != calc and eu(o) < eu(calc)): return "arith"
    return "passed"
for m in MODELS:
    z = tf[(tf.model == m) & (tf.budget == 768)]
    c = np.array([cls_alt(hv, i, K, b) for hv, i, K, b in zip(z.hv, z.scenario_id, z.K, z.chosen_bid)]); T = z.gap.sum()
    R["E4"][m] = {k: float(z.gap.values[c == k].sum() / T) for k in cats}
# ---------------- E5
b0 = tf[tf.budget == 0].set_index(["model", "scenario_id", "repeat_seed"]).chosen_bid
for m in MODELS:
    w = tf[(tf.model == m) & (tf.budget == 768) & ~np.isclose(tf.chosen_bid, tf.calc)]
    same, other, rank_match = [], [], []
    seeds = sorted(tf.repeat_seed.unique())
    pool = tf[(tf.model == m) & (tf.budget == 0)]
    for r in w.itertuples():
        s = S[r.scenario_id]
        same.append(np.isclose(r.chosen_bid, b0[(m, r.scenario_id, r.repeat_seed)]))
        other.append(np.mean([np.isclose(r.chosen_bid, b0[(m, r.scenario_id, s2)]) for s2 in seeds if s2 != r.repeat_seed]))
        rk = sorted(s["bids"]).index(float(r.chosen_bid))
        cand = pool[(pool.K == r.K) & (pool.scenario_id != r.scenario_id)].sample(20, random_state=int(r.scenario_id) % 2**31)
        rank_match.append(np.mean([sorted(S[i]["bids"]).index(float(b)) == rk for i, b in zip(cand.scenario_id, cand.chosen_bid)]))
    R["E5"][m] = dict(repeat_same_order=float(np.mean(same)), repeat_other_order=float(np.mean(other)), bid_rank_baseline=float(np.mean(rank_match)))
# ---------------- E6
k512 = tf[tf.budget == 512].set_index(["model", "scenario_id", "repeat_seed"])
for m in MODELS:
    rows = []
    z = tf[(tf.model == m) & (tf.budget == 768)]
    for r in z.itertuples():
        key = (m, r.scenario_id, r.repeat_seed)
        if key not in k512.index: continue
        a = k512.loc[key]; hv5 = a.hv; c = r.calc
        if c not in hv5 or any(hv5[o][0] >= hv5[c][0] for o in hv5 if o != c) or len(hv5) >= r.K: continue
        rows.append(dict(right512=int(np.isclose(a.chosen_bid, c)), right768=int(np.isclose(r.chosen_bid, c)), complete768=int(len(r.hv) >= r.K)))
    x = pd.DataFrame(rows)
    R["E6"][m] = dict(n=int(len(x)), right512=float(x.right512.mean()), right768_if_complete=float(x[x.complete768 == 1].right768.mean()) if (x.complete768 == 1).any() else None,
                      right768_if_still_incomplete=float(x[x.complete768 == 0].right768.mean()) if (x.complete768 == 0).any() else None,
                      share_completed_by768=float(x.complete768.mean()))
# ---------------- E7
st = d[d.decoding == "stochastic"]
gg = d[d.decoding == "greedy"]
for m in MODELS:
    x = st[st.model == m]; out = {}
    for bud in (128, 384, 768):
        y = x[x.budget == bud]; per_seed = y.groupby("repeat_seed").test_utility_ratio.mean()
        out[str(bud)] = dict(q=float(y.test_utility_ratio.mean()), seed_sd=float(per_seed.std(ddof=1)), energy_j=float(y.energy_joules.mean()))
    y = x[x.budget == 768]; P = y.pivot(index="scenario_id", columns="repeat_seed", values="chosen_bid"); PR = y.pivot(index="scenario_id", columns="repeat_seed", values="constrained_option_probability")
    def vote(bs, ps):
        cnt = {}
        for b_, p_ in zip(bs, ps): c0, s0 = cnt.get(b_, (0, 0.0)); cnt[b_] = (c0 + 1, s0 + p_)
        return max(cnt.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
    out["vote5_768_q"] = float(np.mean([q_of(i, vote(P.loc[i].values, PR.loc[i].values)) for i in P.index]))
    g0 = gg[(gg.model == m) & (gg.budget == 0)].energy_joules.mean(); g7 = gg[(gg.model == m) & (gg.budget == 768)].energy_joules.mean()
    dq = gg[(gg.model == m) & (gg.budget == 768)].test_utility_ratio.mean() - gg[(gg.model == m) & (gg.budget == 0)].test_utility_ratio.mean()
    out["greedy_energy_0_j"] = float(g0); out["greedy_energy_768_j"] = float(g7); out["extra_kj_per_0.01q"] = float((g7 - g0) / (100 * dq) / 1000)
    R["E7"][m] = out
json.dump(R, open(f"{OUTDIR}/t13_judge10.json", "w"), indent=1, default=float)
r3 = lambda o: round(float(o), 3)
for k in R:
    print("=====", k)
    for m, v in R[k].items(): print(m, json.dumps(v, default=r3))
