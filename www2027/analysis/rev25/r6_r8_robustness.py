"""Items 6-8.
6  Landscape noise. (a) Resample the construction auctions within product (1,000 draws), rebuild p_hat, let the
   calculator pick argmax p_hat*(v - b), score on the held-out curve. (b) Resample the held-out auctions within
   product, rebuild the scoring curve, re-score every saved answer and the calculator. (c) Both at once. Report the
   calculator's lead over each model at 768 tokens, gap recovered, and how stable the model ranking is.
7  Hierarchical bootstrap (products resampled, then value families within them) for Table 1's answer column and
   Table 2's shares, next to the paper's family bootstrap.
8  Break-even against the distribution of the best expected profit per decision (held-out), with the paper's c768
   and the 4,096-token run's cost."""
import sys; sys.path.insert(0, "/tmp/www2027/analysis/rev25")
from r_common import *
from scipy.stats import kendalltau
AU = pd.read_csv("/tmp/www2027/analysis/auction_splits.csv")
AU["product"] = AU["item"].map(lambda it: next(p for p in PRODUCTS if p.split()[0].lower() in it.lower()))
TR = {p: AU[(AU["product"] == p) & (AU.split == "train")].final_price.values for p in PRODUCTS}
TE = {p: AU[(AU["product"] == p) & (AU.split != "train")].final_price.values for p in PRODUCTS}
# check that the saved held-out probabilities are the share of held-out prices <= b
chk = [abs(np.mean(TE[S[i]["product"]] <= b) - HELD[i][float(b)]["test_p"]) for i in list(S)[:30] for b in S[i]["bids"]]
assert max(chk) < 1e-9, max(chk)
chk2 = [abs(round(np.mean(TR[S[i]["product"]] <= b), 4) - S[i]["dp"][k]) for i in list(S)[:30] for k, b in enumerate(S[i]["bids"])]
z768 = {m: tf[(tf.model == m) & (tf.budget == 768)].sort_values(["scenario_id", "repeat_seed"]) for m in MODELS}
z0 = {m: tf[(tf.model == m) & (tf.budget == 0)].sort_values(["scenario_id", "repeat_seed"]) for m in MODELS}
sids = z768[MODELS[0]].scenario_id.values
R = {"train_curve_check_max_abs": float(max(chk2))}
def score_all(ptest, calc_choice):
    """ptest: product -> held-out prices; calc_choice: sid -> bid. Returns q768, q0, qcalc per model."""
    out = {}
    U = {}
    for i in S:
        x = ptest[S[i]["product"]]; p = np.array([np.mean(x <= b) for b in S[i]["bids"]]); u = p * (S[i]["v"] - S[i]["bids"])
        U[i] = (dict(zip(S[i]["bids"], u)), u.max())
    qf = lambda i, b: U[i][0][float(b)] / U[i][1] if U[i][1] > 0 else 1.0
    qc = np.array([qf(i, calc_choice[i]) for i in sids]).mean()
    for m in MODELS:
        q7 = np.mean([qf(i, b) for i, b in zip(z768[m].scenario_id, z768[m].chosen_bid)])
        q0_ = np.mean([qf(i, b) for i, b in zip(z0[m].scenario_id, z0[m].chosen_bid)])
        out[m] = (q7, q0_, qc)
    return out
def calc_from(ptrain):
    c = {}
    for i in S:
        x = ptrain[S[i]["product"]]; p = np.round(np.array([np.mean(x <= b) for b in S[i]["bids"]]), 4)
        c[i] = S[i]["bids"][np.argmax(p * (S[i]["v"] - S[i]["bids"]))]
    return c
base_calc = {i: S[i]["calc"] for i in S}
base = score_all(TE, base_calc)
rng = np.random.default_rng(20261114); B = 1000
draws = {k: [] for k in ("train", "test", "both")}
for _ in range(B):
    trs = {p: rng.choice(TR[p], len(TR[p])) for p in PRODUCTS}; tes = {p: rng.choice(TE[p], len(TE[p])) for p in PRODUCTS}
    cc = calc_from(trs)
    draws["train"].append(score_all(TE, cc)); draws["test"].append(score_all(tes, base_calc)); draws["both"].append(score_all(tes, cc))
base_rank = sorted(MODELS, key=lambda m: -base[m][0])
for k, L in draws.items():
    res = {}
    for m in MODELS:
        lead = np.array([d_[m][2] - d_[m][0] for d_ in L]); gr = np.array([(d_[m][0] - d_[m][1]) / (d_[m][2] - d_[m][1]) for d_ in L])
        res[m] = dict(calc_lead_768=dict(pt=float(base[m][2] - base[m][0]), ci=[float(np.percentile(lead, 2.5)), float(np.percentile(lead, 97.5))], share_positive=float((lead > 0).mean())),
                      gap768=dict(pt=float((base[m][0] - base[m][1]) / (base[m][2] - base[m][1])), ci=[float(np.percentile(gr, 2.5)), float(np.percentile(gr, 97.5))]))
    qcs = np.array([L_[MODELS[0]][2] for L_ in L])
    taus = [kendalltau([base[m][0] for m in MODELS], [d_[m][0] for m in MODELS])[0] for d_ in L]
    same = [sorted(MODELS, key=lambda m: -d_[m][0]) == base_rank for d_ in L]
    top = [sorted(MODELS, key=lambda m: -d_[m][0])[0] == base_rank[0] and sorted(MODELS, key=lambda m: -d_[m][0])[-1] == base_rank[-1] for d_ in L]
    R[f"landscape_{k}"] = dict(models=res, qcalc=dict(pt=float(base[MODELS[0]][2]), ci=[float(np.percentile(qcs, 2.5)), float(np.percentile(qcs, 97.5))]),
                              kendall_tau_median=float(np.median(taus)), kendall_tau_p5=float(np.percentile(taus, 5)), same_full_ranking=float(np.mean(same)), same_best_and_worst=float(np.mean(top)))
R["base_ranking_q768"] = [LAB[m] for m in base_rank]
# ---- 7 hierarchical bootstrap
R["hier"] = {}
for m in MODELS:
    z = z768[m].reset_index(drop=True); q0v = z0[m].q_read.values; qv = z.q_read.values; qc = z.qcalc.values; fams = z.fam.values; gap = z.gap.values; cl = z.cls.values
    fn = lambda ids: (qv[ids].mean() - q0v[ids].mean()) / (qc[ids].mean() - q0v[ids].mean())
    sh = lambda ids: np.array([gap[ids][cl[ids] == k].sum() / gap[ids].sum() for k in CATS])
    R["hier"][m] = dict(gap768_family=fboot(fn, fams)[0], gap768_hier=hboot(fn, fams)[0],
                        split_family=dict(zip(CATS, fboot(sh, fams))), split_hier=dict(zip(CATS, hboot(sh, fams))))
# ---- 8 break-even against the distribution of best expected profit
ust = np.array([S[i]["ustar"] for i in sids])
t7 = json.load(open("/tmp/www2027/analysis/revision/out/t7_costs.json"))
r1 = json.load(open("/tmp/www2027/results/followup/run1_summary.json"))
R["ustar_quantiles"] = {q: float(np.percentile(ust, q)) for q in (0, 1, 5, 10, 25, 50, 75, 90, 100)}
R["breakeven"] = {}
for m in MODELS:
    c = t7[LAB[m]]["cost_768_usd"]; be = t7[LAB[m]]["breakeven_usd"]
    z = z768[m]; gain = np.array([S[i]["ustar"] for i in z.scenario_id]) * (z.q_read.values - z0[m].q_read.values)
    out = dict(cost768_usd=c, breakeven_usd=be, ustar_over_be_min=float(ust.min() / be), ustar_over_be_median=float(np.median(ust) / be),
               mean_gain_usd=float(gain.mean()), gain_over_cost=float(gain.mean() / c), share_decisions_ustar_above_be=float((ust > be).mean()))
    if m in r1:
        cn = r1[m]["cost_basis"]["cnat_cents"] / 100; ben = r1[m]["breakeven_natural_usd"]
        out.update(natural_cost_usd=cn, natural_breakeven_usd=ben, natural_ustar_over_be_median=float(np.median(ust) / ben))
    R["breakeven"][m] = out
dump25("r6_r8_robustness", R)
f = lambda x: f"{100*x['pt']:.0f} [{100*x['ci'][0]:.0f}, {100*x['ci'][1]:.0f}]"
print("train curve check", R["train_curve_check_max_abs"], "base ranking", R["base_ranking_q768"])
for k in ("train", "test", "both"):
    L = R[f"landscape_{k}"]; print(f"== landscape {k}: qcalc {L['qcalc']['pt']:.3f} [{L['qcalc']['ci'][0]:.3f}, {L['qcalc']['ci'][1]:.3f}] tau med {L['kendall_tau_median']:.2f} p5 {L['kendall_tau_p5']:.2f} same ranking {L['same_full_ranking']:.2f} same best&worst {L['same_best_and_worst']:.2f}")
    for m in MODELS: print("   ", LAB[m], "lead", f"{L['models'][m]['calc_lead_768']['pt']:.3f} [{L['models'][m]['calc_lead_768']['ci'][0]:.3f}, {L['models'][m]['calc_lead_768']['ci'][1]:.3f}] pos {L['models'][m]['calc_lead_768']['share_positive']:.3f}", "gap", f(L['models'][m]['gap768']))
for m in MODELS:
    h = R["hier"][m]; print(LAB[m], "gap fam", f(h["gap768_family"]), "hier", f(h["gap768_hier"]), "| split hier", {k: f(v) for k, v in h["split_hier"].items()})
print("ustar quantiles", {k: round(v, 2) for k, v in R["ustar_quantiles"].items()})
for m in MODELS: print(LAB[m], {k: round(v, 4) for k, v in R["breakeven"][m].items()})
