"""CPU-only analyses for the WWW 2027 short paper.

Inputs (saved outputs, no new inference):
  saved_records.csv.gz  per-cell decisions
  scenarios.csv         candidate sets with construction (train) and held-out (test) utility
  runtime_cost_recomputed.csv  per-token dollar cost (RTX 4090 at $0.74/h)

Outputs: www_results.json and CSVs in ./out
"""
import json, sys
import numpy as np, pandas as pd

rng = np.random.default_rng(20261105)
OUT = "out"
import os; os.makedirs(OUT, exist_ok=True)

d = pd.read_csv("saved_records.csv.gz")
sc = pd.read_csv("scenarios.csv")
cost = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "runtime_cost_recomputed.csv")
lam = dict(zip(cost.model, cost.lambda_usd_per_token))
label = dict(zip(cost.model, cost.model_label))

# ---- candidate lookup: train/test utility by (scenario, bid)
look = {}
calc = {}
fam = {}
for r in sc.itertuples():
    cands = json.loads(r.candidates_json)
    m = {}
    for c in cands:
        m[float(c["bid"])] = (c["train_u"], c["test_u"])
    look[r.scenario_id] = m
    best_train = max(cands, key=lambda c: c["train_u"])
    calc[r.scenario_id] = best_train["test_u"]
    fam[r.scenario_id] = f"{r.category}|{r.value}"
d["train_utility"] = [look[s][float(b)][0] for s, b in zip(d.scenario_id, d.chosen_bid)]
d["calc_utility"] = d.scenario_id.map(calc)
d["family"] = d.scenario_id.map(fam)

g = d[d.decoding == "greedy"].copy()
BUD = [0, 64, 128, 256, 512, 768]
models = sorted(g.model.unique(), key=lambda m: list(label).index(m) if m in label else 99)

# scenario-level order-averaged table: (model, scenario, budget) -> mean over 3 orders
cell = (g[g.budget >= 0]
        .groupby(["model", "scenario_id", "budget"])
        .agg(U=("test_utility", "mean"), Q=("test_utility_ratio", "mean"),
             Ustar=("test_optimal_utility", "first"), Ucalc=("calc_utility", "first"),
             fam=("family", "first"))
        .reset_index())

res = {"models": {}}
rows_marg = []
for m in models:
    c = cell[cell.model == m]
    piv_U = c.pivot(index="scenario_id", columns="budget", values="U")
    piv_Q = c.pivot(index="scenario_id", columns="budget", values="Q")
    base = c[c.budget == 0].set_index("scenario_id")
    Ustar, Ucalc, famv = base.Ustar, base.Ucalc, base.fam
    Qcalc = Ucalc / Ustar
    U0, U768 = piv_U[0].mean(), piv_U[768].mean()
    Q0, Q768 = piv_Q[0].mean(), piv_Q[768].mean()
    gap_closed_profit = (U768 - U0) / (Ucalc.mean() - U0)
    gap_closed_ratio = (Q768 - Q0) / (Qcalc.mean() - Q0)
    l = lam.get(m)
    l = None if (l is None or (isinstance(l, float) and np.isnan(l))) else l
    cost768 = 768 * l if l else None
    gain = U768 - U0
    # family bootstrap (within product) for gain, gap closed, value/cost
    fams = famv.groupby(famv).groups  # fam -> scenario ids
    prods = {}
    for f, ids in fams.items():
        prods.setdefault(f.split("|")[0], []).append(list(ids))
    bs_gain, bs_gap = [], []
    dU = (piv_U[768] - piv_U[0])
    for _ in range(4000):
        ids = []
        for p, fl in prods.items():
            pick = rng.integers(0, len(fl), len(fl))
            for k in pick: ids += fl[k]
        gg = dU.loc[ids].mean()
        bs_gain.append(gg)
        bs_gap.append(gg / (Ucalc.loc[ids].mean() - piv_U[0].loc[ids].mean()))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    # share of scenarios where 0->768 change exceeds compute cost
    share_pos = float((dU > (cost768 or 0)).mean())
    share_neg = float((dU < 0).mean())
    # marginal value per token on each step
    for a, b in zip(BUD[:-1], BUD[1:]):
        mv = (piv_U[b].mean() - piv_U[a].mean()) / (b - a)
        rows_marg.append(dict(model=m, label=label.get(m, m), step=f"{a}-{b}",
                              usd_per_token=mv, lambda_usd_per_token=l,
                              ratio_to_lambda=(mv / l) if l else None))
    res["models"][m] = dict(
        label=label.get(m, m), U0=U0, U768=U768, Ucalc=float(Ucalc.mean()),
        Ustar=float(Ustar.mean()), Q0=Q0, Q768=Q768, Qcalc=float(Qcalc.mean()),
        profit_gain=gain, profit_gain_ci=ci(bs_gain),
        gap_closed_profit=gap_closed_profit, gap_closed_profit_ci=ci(bs_gap),
        gap_closed_ratio=gap_closed_ratio,
        lambda_usd_per_token=l, cost_768_usd=cost768,
        value_per_compute_dollar=(gain / cost768) if cost768 else None,
        breakeven_stakes_multiplier=(cost768 / gain) if cost768 else None,
        share_scenarios_gain_exceeds_cost=share_pos, share_scenarios_worse=share_neg,
    )

pd.DataFrame(rows_marg).to_csv(f"{OUT}/marginal_value_per_token.csv", index=False)

# ---- confidence-routed adaptive compute: answer at budget 0 if its option
# probability >= tau, otherwise pay for 768 tokens. tau chosen on the
# construction-split utility (train_u), evaluated on held-out utility.
taus = np.round(np.arange(0.30, 1.0001, 0.01), 2)
route_rows, route_sel = [], {}
for m in models:
    gm = g[(g.model == m) & (g.budget.isin([0, 768]))]
    w = gm.pivot_table(index=["scenario_id", "repeat_seed"], columns="budget",
                       values=["test_utility", "train_utility", "constrained_option_probability"])
    p0 = w[("constrained_option_probability", 0)].values
    te0, te768 = w[("test_utility", 0)].values, w[("test_utility", 768)].values
    tr0, tr768 = w[("train_utility", 0)].values, w[("train_utility", 768)].values
    # AUROC of budget-0 probability for "768 changes held-out profit upward"
    # and for budget-0 being held-out optimal
    ustar = gm[gm.budget == 0].set_index(["scenario_id", "repeat_seed"]).test_optimal_utility.reindex(w.index).values
    opt0 = np.isclose(te0, ustar)
    def auroc(score, y):
        y = np.asarray(y, bool); s = np.asarray(score)
        pos, neg = s[y], s[~y]
        if len(pos) == 0 or len(neg) == 0: return None
        r = pd.Series(np.r_[pos, neg]).rank().values
        return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    trg = tr768.mean() - tr0.mean()
    best = None
    for t in taus:
        esc = p0 < t
        te = np.where(esc, te768, te0).mean(); tr = np.where(esc, tr768, tr0).mean()
        tok = 768 * esc.mean()
        route_rows.append(dict(model=m, tau=t, escalate_share=esc.mean(), mean_tokens=tok,
                               heldout_profit=te, construction_profit=tr,
                               heldout_gain_retained=(te - te0.mean()) / (te768.mean() - te0.mean()),
                               construction_gain_retained=(tr - tr0.mean()) / trg))
        # selection rule: smallest escalation share that keeps >=90% of the
        # construction-split gain
        if best is None and (tr - tr0.mean()) / trg >= 0.90:
            pass
    rr = pd.DataFrame([r for r in route_rows if r["model"] == m])
    ok = rr[rr.construction_gain_retained >= 0.90].sort_values("mean_tokens")
    sel = ok.iloc[0]
    # fixed-budget comparison at the same or more tokens
    c = cell[cell.model == m].groupby("budget").U.mean()
    fixed_better = [b for b in BUD if b >= sel.mean_tokens and c[b] >= sel.heldout_profit]
    route_sel[m] = dict(label=label.get(m, m), tau=float(sel.tau),
                        escalate_share=float(sel.escalate_share), mean_tokens=float(sel.mean_tokens),
                        token_saving=float(1 - sel.mean_tokens / 768),
                        heldout_gain_retained=float(sel.heldout_gain_retained),
                        heldout_profit=float(sel.heldout_profit), profit_768=float(te768.mean()),
                        profit_0=float(te0.mean()),
                        cheapest_fixed_budget_matching=(min(fixed_better) if fixed_better else None),
                        auroc_p0_for_budget0_optimal=auroc(p0, opt0))
pd.DataFrame(route_rows).to_csv(f"{OUT}/confidence_routing_curve.csv", index=False)
res["routing"] = route_sel

# ---- accuracy vs utility as a measure of what compute buys: per-model
# exact-accuracy gain vs profit gain (normalized). Shows rank disagreement.
acc = (g[g.budget.isin([0, 768])].assign(opt=lambda x: np.isclose(x.test_utility, x.test_optimal_utility))
       .groupby(["model", "budget"]).opt.mean().unstack())
res["accuracy_gain_pp"] = {m: float(100 * (acc.loc[m, 768] - acc.loc[m, 0])) for m in models}

json.dump(res, open(f"{OUT}/www_results.json", "w"), indent=1, default=float)
for m, v in res["models"].items():
    print(f"{v['label']:12s} gain ${v['profit_gain']:.2f} {np.round(v['profit_gain_ci'],2)} "
          f"gapP {v['gap_closed_profit']:.3f} {np.round(v['gap_closed_profit_ci'],3)} gapQ {v['gap_closed_ratio']:.3f} "
          f"cost {v['cost_768_usd']} val/$ {v['value_per_compute_dollar']} pos {v['share_scenarios_gain_exceeds_cost']:.3f} neg {v['share_scenarios_worse']:.3f} "
          f"U0 {v['U0']:.2f} U768 {v['U768']:.2f} Uc {v['Ucalc']:.2f} U* {v['Ustar']:.2f}")
for m, v in route_sel.items():
    print(v)
print(res["accuracy_gain_pp"])
